"""CPU sampling with a model whose forward has no cache API."""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Model


@torch.inference_mode()
def sample(model, input_ids, max_new_tokens=40, temperature=0.8, seed=7,
           eos_token_id=None):
    rng = torch.Generator(device=input_ids.device).manual_seed(seed)
    ids = input_ids.clone()
    prompt_length = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        probabilities = torch.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probabilities, num_samples=1, generator=rng)
        ids = torch.cat([ids, token], dim=1)
        if eos_token_id is not None and token.item() == eos_token_id:
            break
    return ids[:, prompt_length:]


def main():
    torch.set_num_threads(2)
    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    source = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    model = Model(source).eval()
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a short story about a robot learning to cook."
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    generated = sample(model, input_ids, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
