Project instructions:

This repo is for testing models at numerical questions on tabular datasets. The questions should test not just computing one output, but a full table of outputs, each row of which we should check for correctness.

The processed input datasets fall in the range of 750-1k rows, 5-15 columns. We currently include the tables in the context window in json format, but this is subject to change.

Programming instructions:

Use the python located in our local .venv. This should also apply when running python commands in bash.
Use dotenv (which is installed to our .venv) because our model provider API keys are in a .env file in the root of this repo