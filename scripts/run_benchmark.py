"""Run benchmark: send each question + dataset to each model via provider SDKs."""

import argparse
import asyncio
import collections
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from config import (
    CONFIGS, SYSTEM_PROMPT, ALL_DATASETS, RATE_LIMITS, OUTPUT_TOKEN_ESTIMATES,
    get_effort,
)
from providers import (
    call_anthropic, call_openai, call_gemini, call_provider, call_model,
)

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "prepared"
Q_DIR = BASE_DIR / "questions"
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_dataset_token_cache: dict[str, int] = {}


def estimate_input_tokens(json_data: str, question_text: str) -> int:
    """Estimate input tokens for a single API call using tiktoken if available."""
    cache_key = str(len(json_data))  # same dataset = same length
    if cache_key not in _dataset_token_cache:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            _dataset_token_cache[cache_key] = len(enc.encode(json_data))
        except ImportError:
            # Fallback: JSON tokenizes at ~0.42 tokens/byte based on our datasets
            _dataset_token_cache[cache_key] = int(len(json_data) * 0.42)
    return _dataset_token_cache[cache_key] + len(question_text) // 4 + 70


# ---------------------------------------------------------------------------
# Rate limiting: token-aware budget per provider
# ---------------------------------------------------------------------------

class TokenBudget:
    """Sliding-window (60s) token-per-minute and request-per-minute tracker."""

    def __init__(self, tpm_limit: int, rpm_limit: int | None = None, safety: float = 0.9):
        self.tpm_limit = int(tpm_limit * safety)
        self.rpm_limit = int(rpm_limit * safety) if rpm_limit else None
        self._records: collections.deque = collections.deque()
        self._lock = asyncio.Lock()

    def _purge_old(self):
        cutoff = time.monotonic() - 60
        while self._records and self._records[0][0] < cutoff:
            self._records.popleft()

    def tokens_used(self) -> int:
        self._purge_old()
        return sum(t for _, t in self._records)

    def requests_used(self) -> int:
        self._purge_old()
        return len(self._records)

    async def wait_for(self, estimated_tokens: int):
        """Block until estimated tokens fit within the TPM (and RPM) budget, then reserve."""
        while True:
            async with self._lock:
                self._purge_old()
                tokens_ok = self.tokens_used() + estimated_tokens <= self.tpm_limit
                rpm_ok = self.rpm_limit is None or self.requests_used() < self.rpm_limit
                if tokens_ok and rpm_ok:
                    self._records.append((time.monotonic(), estimated_tokens))
                    return
            await asyncio.sleep(1.0)

    def record_actual(self, estimated: int, actual: int):
        """Correct the most recent matching reservation with actual usage."""
        for i in range(len(self._records) - 1, -1, -1):
            if self._records[i][1] == estimated:
                self._records[i] = (self._records[i][0], actual)
                break


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def load_completed(results_file):
    """Load already-completed (question_id, model_name) pairs from JSONL."""
    completed = set()
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            completed.add((rec["question_id"], rec["model_name"]))
    return completed


def extract_answer(response_text):
    """Extract the answer after 'ANSWER:' marker."""
    if "ANSWER:" in response_text:
        return response_text.split("ANSWER:")[-1].strip()
    return response_text.strip().split("\n")[-1].strip()


# ---------------------------------------------------------------------------
# Task tracker & Rich TUI
# ---------------------------------------------------------------------------

class TaskTracker:
    """Track status of all benchmark tasks for live TUI display."""

    MAX_ERRORS = 10  # Keep last N errors for display

    def __init__(self, error_log_path=None):
        self.tasks: dict[tuple[str, str], dict] = {}
        self.start_time = time.time()
        self.recent_errors: list[dict] = []  # [{run_name, question_id, error}]
        self.error_log_path = error_log_path

    def register(self, question_id, run_name, dataset):
        self.tasks[(question_id, run_name)] = {
            "dataset": dataset,
            "status": "pending",
            "elapsed": None,
        }

    def set_skipped(self, question_id, run_name, dataset):
        self.tasks[(question_id, run_name)] = {
            "dataset": dataset,
            "status": "skipped",
            "elapsed": None,
        }

    def set_running(self, question_id, run_name):
        t = self.tasks.get((question_id, run_name))
        if t:
            t["status"] = "running"

    def set_done(self, question_id, run_name, elapsed):
        t = self.tasks.get((question_id, run_name))
        if t:
            t["status"] = "done"
            t["elapsed"] = elapsed

    def set_failed(self, question_id, run_name, error=None):
        t = self.tasks.get((question_id, run_name))
        if t:
            t["status"] = "failed"
        if error:
            entry = {
                "run_name": run_name,
                "question_id": question_id,
                "error": error,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self.recent_errors.append(entry)
            if len(self.recent_errors) > self.MAX_ERRORS:
                self.recent_errors.pop(0)
            if self.error_log_path:
                with open(self.error_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

    def counts(self, run_name=None, dataset=None):
        """Return dict of status -> count, optionally filtered."""
        c = {"pending": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
        for (qid, rn), t in self.tasks.items():
            if run_name and rn != run_name:
                continue
            if dataset and t["dataset"] != dataset:
                continue
            c[t["status"]] += 1
        return c


def _fmt_cell(counts):
    """Format a cell showing status counts with color."""
    d = counts["done"] + counts["skipped"]
    r, f, p = counts["running"], counts["failed"], counts["pending"]
    total = d + r + f + p
    if total == 0:
        return Text("-", style="dim")

    text = Text()
    if d > 0:
        text.append(f"{d}", style="bold green")
        if r or f or p:
            text.append("/")
    if r > 0:
        text.append(f"{r}", style="bold yellow")
        if f or p:
            text.append("/")
    if f > 0:
        text.append(f"{f}", style="bold red")
        if p:
            text.append("/")
    if p > 0:
        text.append(f"{p}", style="dim")

    return text


def build_table(tracker, run_names, datasets, static=False):
    """Build the Rich table showing benchmark progress."""
    elapsed = time.time() - tracker.start_time
    total_counts = tracker.counts()
    done_count = total_counts["done"] + total_counts["skipped"]
    all_count = sum(total_counts.values())

    if all_count > 0:
        pct = done_count / all_count * 100
    else:
        pct = 0

    mins, secs = divmod(int(elapsed), 60)
    if static:
        title = f"Calcubench — {done_count}/{all_count} done ({pct:.0f}%)"
    else:
        title = f"Calcubench — {done_count}/{all_count} done ({pct:.0f}%) — {mins}m {secs:02d}s"

    # Column header: legend
    table = Table(title=title, title_style="bold", expand=False, padding=(0, 1))
    table.add_column("Config", style="bold", no_wrap=True)
    for ds in datasets:
        # Shorten dataset names for column headers
        label = ds.replace("census_", "cen_").replace("southeast", "se").replace("national", "nat")
        table.add_column(label, justify="center")
    table.add_column("Total", justify="center", style="bold")

    # One row per run config
    for rn in run_names:
        row = [rn]
        rn_total = tracker.counts(run_name=rn)
        for ds in datasets:
            c = tracker.counts(run_name=rn, dataset=ds)
            row.append(_fmt_cell(c))
        # Total column: done/all for this config
        rn_all = sum(rn_total.values())
        rn_done = rn_total["done"] + rn_total["skipped"]
        total_text = Text(f"{rn_done}/{rn_all}")
        if rn_done == rn_all and rn_all > 0:
            total_text.stylize("bold green")
        row.append(total_text)
        table.add_row(*row)

    # Footer: totals per dataset
    table.add_section()
    footer = [Text("Total", style="bold")]
    for ds in datasets:
        c = tracker.counts(dataset=ds)
        footer.append(_fmt_cell(c))
    total_text = Text(f"{done_count}/{all_count}")
    if done_count == all_count and all_count > 0:
        total_text.stylize("bold green")
    footer.append(total_text)
    table.add_row(*footer)

    # Legend
    if static:
        table.caption = "[green]done[/]/[dim]pending[/]"
    else:
        table.caption = "[green]done[/]/[yellow]running[/]/[red]failed[/]/[dim]pending[/]"

    # Error log below the table
    if not tracker.recent_errors:
        return table

    error_lines = Text()
    error_lines.append("\nRecent errors:\n", style="bold red")
    for err in tracker.recent_errors:
        error_lines.append(f"  {err['run_name']} ", style="bold")
        error_lines.append(f"({err['question_id']})", style="dim")
        # Truncate long error messages
        msg = err["error"]
        if len(msg) > 120:
            msg = msg[:120] + "..."
        error_lines.append(f" {msg}\n", style="red")

    return Group(table, error_lines)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def process_one(run_name, model_cfg, q, ds_name, json_data,
                      semaphore, budget, results_file, file_lock, completed, tracker):
    """Handle one (question, model_config) pair: call model, extract, write result."""
    # Wait for token budget before proceeding
    effort = get_effort(run_name)
    input_est = estimate_input_tokens(json_data, q["question"])
    output_est = OUTPUT_TOKEN_ESTIMATES.get(effort, OUTPUT_TOKEN_ESTIMATES["default"])
    total_est = input_est + output_est
    await budget.wait_for(total_est)

    tracker.set_running(q["id"], run_name)
    t0 = time.time()

    result_tuple = await call_model(
        run_name, model_cfg, q["question"], json_data, semaphore,
        answer_type=q.get("answer_type"),
    )
    if result_tuple is None:
        tracker.set_failed(q["id"], run_name)
        return
    # call_model returns (None, error_string) on failure
    if result_tuple[0] is None:
        tracker.set_failed(q["id"], run_name, error=result_tuple[1])
        return

    response, reasoning, usage = result_tuple
    # Correct the token budget reservation with actual usage
    actual_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    if actual_tokens > 0:
        budget.record_actual(total_est, actual_tokens)

    extracted = extract_answer(response)
    elapsed = round(time.time() - t0, 2)
    tracker.set_done(q["id"], run_name, elapsed)
    result = {
        "question_id": q["id"],
        "dataset": ds_name,
        "model_name": run_name,
        "model_id": model_cfg["model"],
        "reasoning_effort": effort,
        "question": q["question"],
        "expected_answer": q["answer"],
        "answer_type": q["answer_type"],
        "tolerance": q["tolerance"],
        "raw_response": response,
        "reasoning_trace": reasoning,
        "extracted_answer": extracted,
        "usage": usage,
        "elapsed_seconds": elapsed,
    }
    if "value_types" in q:
        result["value_types"] = q["value_types"]
    if "tolerances" in q:
        result["tolerances"] = q["tolerances"]

    async with file_lock:
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        completed.add((q["id"], run_name))


async def run(run_names, datasets, question_ids=None, concurrency=None, max_questions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "results.jsonl"
    completed = load_completed(results_file)

    # Per-provider semaphores for rate limiting
    # Tier 3 Anthropic (2,000 RPM), Tier 4 OpenAI (10,000 RPM), Tier 1 Gemini (~150 RPM)
    default_concurrency = {
        "anthropic": 8,
        "openai": 15,
        "gemini": 2,
    }
    provider_semaphores: dict[str, asyncio.Semaphore] = {}
    for run_name in run_names:
        provider = CONFIGS[run_name]["provider"]
        if provider not in provider_semaphores:
            if concurrency is None:
                c = default_concurrency.get(provider, 5)
            elif isinstance(concurrency, dict):
                c = concurrency.get(provider, default_concurrency.get(provider, 5))
            else:
                c = concurrency
            provider_semaphores[provider] = asyncio.Semaphore(c)

    # Token budgets per provider (sliding-window TPM/RPM tracking)
    provider_budgets: dict[str, TokenBudget] = {}
    for provider, limits in RATE_LIMITS.items():
        if provider in provider_semaphores:
            provider_budgets[provider] = TokenBudget(
                tpm_limit=limits["tpm"],
                rpm_limit=limits.get("rpm"),
            )

    file_lock = asyncio.Lock()
    error_log = RESULTS_DIR / "errors.jsonl"
    tracker = TaskTracker(error_log_path=error_log)

    # Collect all tasks
    tasks = []
    for ds_name in datasets:
        data_path = DATA_DIR / f"{ds_name}.json"
        if not data_path.exists():
            continue
        json_data = data_path.read_text()
        q_path = Q_DIR / f"{ds_name}.json"
        if not q_path.exists():
            continue
        questions = json.loads(q_path.read_text())

        if question_ids:
            questions = [q for q in questions if q["id"] in question_ids]
            if not questions:
                continue

        if max_questions is not None:
            questions = questions[:max_questions]

        for q in questions:
            for run_name in run_names:
                key = (q["id"], run_name)
                if key in completed:
                    tracker.set_skipped(q["id"], run_name, ds_name)
                    continue

                tracker.register(q["id"], run_name, ds_name)
                model_cfg = CONFIGS[run_name]
                provider = model_cfg["provider"]
                semaphore = provider_semaphores[provider]
                budget = provider_budgets[provider]

                tasks.append(
                    process_one(
                        run_name, model_cfg, q, ds_name, json_data,
                        semaphore, budget, results_file, file_lock, completed, tracker,
                    )
                )

    if not tasks:
        print("All tasks already completed.")
        return

    sem_info = {p: s._value for p, s in provider_semaphores.items()}
    print(f"Dispatching {len(tasks)} API calls (concurrency: {sem_info})\n")

    with Live(build_table(tracker, run_names, datasets), refresh_per_second=4, transient=False) as live:
        async def refresh_loop():
            try:
                while True:
                    await asyncio.sleep(0.25)
                    live.update(build_table(tracker, run_names, datasets))
            except asyncio.CancelledError:
                pass

        refresh_task = asyncio.create_task(refresh_loop())
        await asyncio.gather(*tasks)
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        # Final render
        live.update(build_table(tracker, run_names, datasets))

    print("\nAll tasks finished.")


def show_status(run_names, datasets, question_ids=None, max_questions=None):
    """Print a static snapshot of benchmark progress from existing results."""
    from rich.console import Console

    results_file = RESULTS_DIR / "results.jsonl"
    completed = load_completed(results_file)

    tracker = TaskTracker()

    for ds_name in datasets:
        q_path = Q_DIR / f"{ds_name}.json"
        if not q_path.exists():
            continue
        questions = json.loads(q_path.read_text())

        if question_ids:
            questions = [q for q in questions if q["id"] in question_ids]

        if max_questions is not None:
            questions = questions[:max_questions]

        for q in questions:
            for run_name in run_names:
                key = (q["id"], run_name)
                if key in completed:
                    tracker.set_skipped(q["id"], run_name, ds_name)
                else:
                    tracker.register(q["id"], run_name, ds_name)

    # Load elapsed times from results for done tasks
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            qid, rn = rec["question_id"], rec["model_name"]
            if rn in run_names and (qid, rn) in tracker.tasks:
                tracker.tasks[(qid, rn)]["status"] = "done"
                tracker.tasks[(qid, rn)]["elapsed"] = rec.get("elapsed_seconds")

    console = Console()
    console.print(build_table(tracker, run_names, datasets, static=True))

    # Summary counts
    counts = tracker.counts()
    done = counts["done"] + counts["skipped"]
    total = sum(counts.values())
    pending = counts["pending"]
    console.print(f"\n[bold]{done}[/bold]/{total} completed, [dim]{pending} pending[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Run Calcubench benchmark")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print a snapshot of current progress without running any API calls",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Models to test (model@effort)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        help="Datasets to test",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="Only run specific question IDs (e.g. imdb_002 census_nat_001)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Only run the first N questions from each dataset/benchmark",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help="Concurrent API calls per provider. Single int (e.g. 10) applies to all, "
             "or comma-separated provider=N pairs (e.g. anthropic=20,openai=20,gemini=5). "
             "Defaults: anthropic=20, openai=20, gemini=5",
    )
    args = parser.parse_args()

    if args.status:
        show_status(args.models, args.datasets, question_ids=args.questions,
                    max_questions=args.max_questions)
        return

    # Parse concurrency arg
    concurrency = None
    if args.concurrency:
        if "=" in args.concurrency:
            concurrency = {}
            for pair in args.concurrency.split(","):
                provider, n = pair.split("=")
                concurrency[provider.strip()] = int(n.strip())
        else:
            concurrency = int(args.concurrency)

    print(f"Models ({len(args.models)}):")
    for name in args.models:
        print(f"  {name} -> {CONFIGS[name]['model']}")

    asyncio.run(run(args.models, args.datasets,
                     question_ids=args.questions,
                     concurrency=concurrency,
                     max_questions=args.max_questions))


if __name__ == "__main__":
    main()
