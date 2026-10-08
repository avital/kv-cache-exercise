# KV cache exercise

Add KV caching to a working CPU sampler. It uses SmolLM2-135M-Instruct weights
with explicit attention in [model.py](model.py). The sampler and its checks are in
[main.py](main.py).

## Run

Python 3.10+. No GPU needed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The first run downloads the model and tokenizer. It samples 64 tokens from a
fixed prompt, prints the text and per-token times, then runs the checks below.
The starter passes the text check and fails the timing check. That is expected
until you implement KV caching.

## Task

**Write the code yourself, without AI assistance.** You can use AI to learn
about KV caching, but do not use it to write the implementation.

Edit `model.py` and the sampling loop in `main.py`:

- Process the prompt once, then only the new token at each sampling step.
- Implement the cache inside attention. Do not use Hugging Face's built-in cache.
- Keep logits equivalent to a full forward on the same token history.

Choose the cache representation, tensor layout, and internal API yourself.
Keep the prompt, sampling settings, and checks in `main.py` unchanged.

## Checks

`python main.py` runs both checks using the same pretrained model as the sampler:

- **Text:** Sample 64 tokens at temperature 0.8 with seed 7, then decode them.
  The text must exactly match the hardcoded reference string in `main.py`.
- **Time per token:** Record 64 timings. Each time for tokens 2–64 must be
  positive and less than token 1's time. Model loading and a two-token warmup
  are excluded.

Nothing inside the cache is inspected. CPU timings can fluctuate on a busy
machine. Passing checks this fixed example, not every possible input.

## Solution

[diff](instructor/add_kv_cache.diff)
