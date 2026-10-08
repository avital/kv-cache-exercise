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
- Handle repeated sampling requests without carrying over state.

Choose the cache representation, tensor layout, and internal API yourself.
Keep the existing `sample(...)` entry point so the tests can call it.

Scope: one unpadded sequence, float32, inference only.

## Test

Tests use tiny random weights on CPU. No model download.

```sh
python -m pytest -q tests -k 'uncached or reproducible'  # Check the starter
python -m pytest -q tests                             # Grade your implementation
```

Cache-specific tests fail until you implement the task. They compare predictions
against a full forward and check how many tokens enter the projections.

Solution: [diff](instructor/add_kv_cache.diff)
