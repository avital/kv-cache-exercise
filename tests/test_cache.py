"""CPU grader: tiny real attention, random weights, no model download.

Default: test the student's model.py and sample.py. To test the reference:
    CANDIDATE_DIR=instructor pytest -q tests
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

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


def trace_sample(model, prompt, steps, forced=None, temperature=0.8):
    # Observe predictions at the existing output projection, so the learner
    # can choose the forward signature, return value, and cache representation.
    records = []
    last_logits = None
    draws = iter(forced) if forced is not None else None
    original_draw = torch.multinomial

    def observe(module, args, output):
        nonlocal last_logits
        last_logits = output.reshape(-1, output.shape[-1])[-1:].detach().clone()

    def draw(probabilities, num_samples, generator):
        assert last_logits is not None
        torch.testing.assert_close(
            probabilities, torch.softmax(last_logits / temperature, dim=-1),
            rtol=1e-5, atol=1e-7,
        )
        records.append(last_logits)
        if draws is not None:
            return torch.tensor([[next(draws)]], device=probabilities.device)
        return original_draw(probabilities, num_samples, generator=generator)

    hook = model.lm_head.register_forward_hook(observe)
    try:
        with patch("torch.multinomial", side_effect=draw):
            generated = sample(model, prompt, max_new_tokens=steps,
                               temperature=temperature, seed=7)
    finally:
        hook.remove()
    return generated, records


@torch.inference_mode()
def assert_predictions_match(source, prompt, generated, records):
    assert len(records) == generated.shape[1]
    for step, actual in enumerate(records):
        history = torch.cat([prompt, generated[:, :step]], dim=1)
        expected = source(history, use_cache=False).logits[:, -1, :]
        assert_logits_close(actual, expected)


@torch.inference_mode()
def test_uncached_forward_matches_original_model(models):
    model, source = models
    ids = torch.arange(3, 20).unsqueeze(0)
    generated, records = trace_sample(model, ids, 1)
    assert_predictions_match(source, ids, generated, records)


def test_sampling_runs_and_is_reproducible(models):
    model, _ = models
    ids = torch.tensor([[3, 7, 11]])
    first = sample(model, ids, max_new_tokens=8, seed=19)
    second = sample(model, ids, max_new_tokens=8, seed=19)
    assert first.shape == (1, 8)
    assert torch.equal(first, second)


@pytest.mark.parametrize("prompt_length", [1, 7, 31])
def test_cached_logits_with_fixed_continuation(models, prompt_length):
    model, source = models
    prompt = torch.arange(3, 3 + prompt_length).unsqueeze(0)
    forced = [11, 23, 35, 47]
    generated, records = trace_sample(model, prompt, len(forced), forced=forced)
    assert torch.equal(generated, torch.tensor([forced]))
    assert_predictions_match(source, prompt, generated, records)


@pytest.mark.parametrize("prompt_length", [7, 31])
def test_sampler_projects_only_new_tokens(models, prompt_length):
    model, _ = models
    calls, hooks = [], []
    for layer in model.layers:
        for projection in (layer.q, layer.k, layer.v):
            lengths = []
            calls.append(lengths)
            hooks.append(projection.register_forward_pre_hook(
                lambda module, args, lengths=lengths:
                    lengths.append(args[0].numel() // module.in_features)
            ))
    try:
        prompt = torch.arange(3, 3 + prompt_length).unsqueeze(0)
        sample(model, prompt, max_new_tokens=4)
    finally:
        for hook in hooks:
            hook.remove()
    assert all(sum(lengths) == prompt_length + 3 for lengths in calls), calls


def test_sampler_starts_each_request_fresh(models):
    model, source = models
    a, b = torch.tensor([[3, 5]]), torch.tensor([[37, 15, 21]])
    first, first_records = trace_sample(model, a, 6)
    other, other_records = trace_sample(model, b, 6)
    again, again_records = trace_sample(model, a, 6)
    assert torch.equal(first, again)
    for prompt, generated, records in ((a, first, first_records),
                                      (b, other, other_records),
                                      (a, again, again_records)):
        assert_predictions_match(source, prompt, generated, records)
