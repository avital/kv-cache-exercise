"""CPU sampler. Print the result, then check its text and per-token times."""

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
    cache = [None] * len(model.layers)
    step_ids = ids
    for _ in range(max_new_tokens):
        started = perf_counter()
        logits = model(step_ids, cache)[:, -1, :]
        probabilities = torch.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probabilities, num_samples=1, generator=rng)
        ids = torch.cat([ids, token], dim=1)
        step_ids = token
        if timings is not None:
            timings.append(perf_counter() - started)
    return ids[:, prompt_length:]


def main():
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
    timings = []
    generated = sample(model, input_ids, max_new_tokens=64,
                       temperature=0.8, seed=7, timings=timings)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(text)
    print("\ntoken\tms")
    for index, seconds in enumerate(timings, start=1):
        print(f"{index}\t{seconds * 1000:.2f}")

    assert text == (
        'One fateful evening, in a sleek, high-tech laboratory, a robot named '
        'Kae-san, or "Kae," began its remarkable journey. Kae was a distant '
        'cousin of Earth robots, known for their innovative capabilities and '
        "unparalleled precision in their fields, but Kae's true potential lay "
        'in the realm of'
    ), "Sampled text differs from the reference."
    print("\nText check passed.")
    assert len(timings) == 64 and all(
        0 < seconds < timings[0] for seconds in timings[1:]
    ), "Timing check failed: every token after the first must be faster. Implement KV caching."
    print("Timing check passed.")


if __name__ == "__main__":
    main()
