"""CPU sampler. Print the result, then check its text and per-token times."""

import os
import sys
from textwrap import wrap
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Model


@torch.inference_mode()
def sample(model, input_ids, max_new_tokens=64, temperature=0.8, seed=7,
           timings=None):
    rng = torch.Generator(device=input_ids.device).manual_seed(seed)
    ids = input_ids.clone()
    prompt_length = ids.shape[1]
    for _ in range(max_new_tokens):
        started = perf_counter()
        logits = model(ids)[:, -1, :]
        probabilities = torch.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probabilities, num_samples=1, generator=rng)
        ids = torch.cat([ids, token], dim=1)
        if timings is not None:
            timings.append(perf_counter() - started)
    return ids[:, prompt_length:]


def color(text, code):
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        return f"\033[{code}m{text}\033[0m"
    return text


def print_result(text, timings):
    print(color("\n📖 Sampled story", "1;36"))
    print(color("╭" + "─" * 70 + "╮", "36"))
    for line in wrap(text, width=68):
        print(color("│", "36") + f" {line:<68} " + color("│", "36"))
    print(color("╰" + "─" * 70 + "╯", "36"))

    print(color("\n⏱  Time per token", "1;36"))
    print("  Token      ms     Time relative to token 1")
    for index, seconds in enumerate(timings, start=1):
        if index == 1:
            shade, mark = "36", "◆"
        elif 0 < seconds < timings[0]:
            shade, mark = "32", "✓"
        else:
            shade, mark = "31", "✗"
        bar = "━" * min(28, max(1, round(seconds / max(timings[0], 1e-9) * 12)))
        print(color(f"  {index:>3}   {seconds * 1000:>7.2f}   {bar:<28} {mark}", shade))
    print("\n  " + color("◆ first token", "36") + "   "
          + color("✓ faster", "32") + "   " + color("✗ not faster", "31"))


def main():
    print(color("\n🤖 KV cache lab", "1;36"))
    print("Loading SmolLM2 on CPU…", flush=True)
    torch.set_num_threads(2)
    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    source = AutoModelForCausalLM.from_pretrained(
        checkpoint, revision=revision, dtype=torch.float32,
    )
    model = Model(source).eval()
    prompt = "Write a short story about a robot learning to cook."
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    sample(model, input_ids, max_new_tokens=2)  # Warm up CPU kernels.
    print("Sampling 64 tokens…", flush=True)
    timings = []
    generated = sample(model, input_ids, max_new_tokens=64,
                       temperature=0.8, seed=7, timings=timings)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print_result(text, timings)

    assert text == (
        'One fateful evening, in a sleek, high-tech laboratory, a robot named '
        'Kae-san, or "Kae," began its remarkable journey. Kae was a distant '
        'cousin of Earth robots, known for their innovative capabilities and '
        "unparalleled precision in their fields, but Kae's true potential lay "
        'in the realm of'
    ), "Sampled text differs from the reference."
    print(color("\n✅ Text matches the reference.", "1;32"))
    assert len(timings) == 64 and all(
        0 < seconds < timings[0] for seconds in timings[1:]
    ), "Timing check failed: every token after the first must be faster. Implement KV caching."
    print(color("✅ Every later token was faster than token 1.", "1;32"))
    print(color("\n🎉 All checks passed!", "1;32"))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(color(f"\n❌ {error}", "1;31"))
        sys.exit(1)
