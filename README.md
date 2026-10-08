# KV cache exercise

Add KV caching to a working CPU sampler. It uses SmolLM2-135M-Instruct weights
with explicit attention in [model.py](model.py). The sampler and its checks are in
[main.py](main.py).

## Setup

Python 3.10+. No GPU needed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Task

**Write the code yourself, without AI assistance.** You can use AI to learn
about KV caching, but do not use it to write the implementation.

Edit `model.py` and the sampling loop in `main.py`:

- Process the prompt once, then only the new token at each sampling step.
- Implement the cache inside attention. Do not use Hugging Face's built-in cache.
- Keep logits equivalent to a full forward on the same token history.

Choose the cache representation, tensor layout, and internal API yourself.
Keep the prompt, sampling settings, and checks in `main.py` unchanged.

## Tests

```sh
python main.py
```

This is both the demo and the test. It uses SmolLM2 on CPU; the first run
downloads the model and tokenizer. It prints the story and per-token times,
then checks:

- **Text:** Generate 64 tokens from the fixed prompt at temperature 0.8 with
  seed 7. The decoded text must exactly match the hardcoded reference string.
- **Timing:** Record 64 timings. Tokens 2–64 must each take a positive amount
  of time and less time than token 1. Loading and a two-token warmup are excluded.

The starter should pass the text check and fail timing. After adding caching,
both should pass. A failed check exits with status 1.

No cache internals are inspected. CPU timings can fluctuate on a busy machine.

## Solution

[diff](instructor/add_kv_cache.diff)
