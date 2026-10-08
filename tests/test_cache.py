"""CPU grader: tiny real attention, random weights, no model download.

Default: test the student's model.py and sample.py. To test the reference:
    CANDIDATE_DIR=instructor pytest -q tests
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(os.environ.get("CANDIDATE_DIR", ROOT)).resolve()))
from model import Model
from sample import sample


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    checkpoint = os.environ.get("PRETRAINED_MODEL")
    if checkpoint:
        source = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.float32, attn_implementation="eager"
        )
    else:
        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        source = LlamaForCausalLM(config)
    return Model(source).eval(), source.eval()


def assert_logits_close(actual, expected):
    # Large FP32 models accumulate more error when CPU matrix shapes change.
    atol = 1e-3 if os.environ.get("PRETRAINED_MODEL") else 1e-6
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=atol)


@torch.inference_mode()
def test_uncached_forward_matches_original_model(models):
    model, source = models
    ids = torch.arange(3, 20).unsqueeze(0)
    assert_logits_close(model(ids), source(ids, use_cache=False).logits)


def test_sampling_runs_and_is_reproducible(models):
    model, _ = models
    ids = torch.tensor([[3, 7, 11]])
    first = sample(model, ids, max_new_tokens=8, seed=19)
    second = sample(model, ids, max_new_tokens=8, seed=19)
    assert first.shape == (1, 8)
    assert torch.equal(first, second)


@pytest.mark.parametrize("prompt_length", [1, 7, 31])
@torch.inference_mode()
def test_cached_logits_with_fixed_continuation(models, prompt_length):
    model, source = models
    prompt = torch.arange(3, 3 + prompt_length).unsqueeze(0)
    ids = torch.cat([prompt, torch.tensor([[11, 23, 35, 47]])], dim=1)
    cache = [None] * len(model.layers)
    for end in range(prompt_length, ids.shape[1] + 1):
        step_ids = ids[:, :end] if end == prompt_length else ids[:, end - 1:end]
        actual = model(step_ids, cache)[:, -1, :]
        expected = source(ids[:, :end], use_cache=False).logits[:, -1, :]
        assert_logits_close(actual, expected)
        for k, v in cache:
            assert k.shape == v.shape == (1, model.kv_heads, end, model.head_dim)


@torch.inference_mode()
def test_cached_chunks_use_correct_positions_and_causal_mask(models):
    model, source = models
    ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23, 29, 31]])
    cache = [None] * len(model.layers)
    start = 0
    for length in [3, 1, 2, 4]:
        end = start + length
        actual = model(ids[:, start:end], cache)
        expected = source(ids[:, :end], use_cache=False).logits[:, start:end]
        assert_logits_close(actual, expected)
        start = end


@torch.inference_mode()
def test_cache_keeps_old_keys_and_values_unchanged(models):
    model, _ = models
    cache = [None] * len(model.layers)
    model(torch.tensor([[3, 5, 7]]), cache)
    snapshots = [(k.clone(), v.clone()) for k, v in cache]
    model(torch.tensor([[11, 13]]), cache)
    for (k, v), (old_k, old_v) in zip(cache, snapshots):
        assert torch.equal(k[:, :, :3], old_k)
        assert torch.equal(v[:, :, :3], old_v)


def test_sampler_projects_only_new_tokens(models):
    model, _ = models
    calls, hooks = [], []
    for layer in model.layers:
        for projection in (layer.q, layer.k, layer.v):
            lengths = []
            calls.append(lengths)
            hooks.append(projection.register_forward_pre_hook(
                lambda module, args, lengths=lengths: lengths.append(args[0].shape[1])
            ))
    try:
        sample(model, torch.arange(3, 10).unsqueeze(0), max_new_tokens=4)
    finally:
        for hook in hooks:
            hook.remove()
    assert all(lengths == [7, 1, 1, 1] for lengths in calls)


@torch.inference_mode()
def test_interleaved_requests_have_independent_caches(models):
    model, source = models
    a, b = torch.tensor([[3, 5, 7]]), torch.tensor([[19, 23]])
    cache_a, cache_b = [None] * len(model.layers), [None] * len(model.layers)
    model(a, cache_a)
    model(b, cache_b)
    next_a, next_b = torch.tensor([[11]]), torch.tensor([[29]])
    actual_a, actual_b = model(next_a, cache_a), model(next_b, cache_b)
    expected_a = source(torch.cat([a, next_a], dim=1), use_cache=False).logits[:, -1:]
    expected_b = source(torch.cat([b, next_b], dim=1), use_cache=False).logits[:, -1:]
    assert_logits_close(actual_a, expected_a)
    assert_logits_close(actual_b, expected_b)


def test_sampler_starts_each_request_fresh(models):
    model, _ = models
    a, b = torch.tensor([[3, 5]]), torch.tensor([[37, 15, 21]])
    first = sample(model, a, max_new_tokens=6, seed=7)
    sample(model, b, max_new_tokens=6, seed=7)
    again = sample(model, a, max_new_tokens=6, seed=7)
    assert torch.equal(first, again)
