# KV cache exercise

Add KV caching to a working CPU sampler. It uses SmolLM2-135M-Instruct weights
with explicit attention in [model.py](model.py). The decoding loop is in
[sample.py](sample.py).

## Run

Python 3.10+. No GPU needed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sample.py "Write a short story about a robot learning to cook."
```

The first sampling run downloads the model and tokenizer.
Prints the generated text, then the time in milliseconds for each sampled token.
Model loading and an initial warmup are excluded. CPU timings can fluctuate;
look for the overall trend. The default run samples up to 128 tokens.

## Task

Edit `model.py` and `sample.py`:

- Process the prompt once, then only the new token at each sampling step.
- Implement the cache inside attention. Do not use Hugging Face's built-in cache.
- Keep logits equivalent to a full forward on the same token history.

Choose the cache representation, tensor layout, and internal API yourself.
Keep the existing `sample(...)` entry point so the tests can call it.

## Test

Tests use a small model with fixed random weights on CPU. No model download.

```sh
python -m pytest -q tests -k sampled_tokens  # Check the starter's output
python -m pytest -q tests                   # Check your implementation
```

Tests compare sampled token IDs against hardcoded reference outputs for three
prompt/seed pairs. They also check that tokens 2–16 each take less time than token
1, using a 512-token prompt after a warmup. The timing test should fail on the
starter. Nothing inside the cache is inspected.

Timing can fluctuate on a busy CPU. Passing these tests checks the given examples,
not every possible input.

## Solution

[diff](instructor/add_kv_cache.diff)
