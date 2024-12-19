import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


def format_prompt(instruction: str, input_str: str) -> str:
    if input_str.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1500,
    temperature: float = 0.4,
    do_sample: bool = True,
) -> str:
    """
    Generate a response from model given a prompt.
    """
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = input_ids.shape[-1]

    # Generate output
    generation_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )

    # Extract only generated tokens
    generated_tokens = generation_output[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def main():
    # Model path
    model_path = "unboundedmarket/smart_contract_explainer_open_llama_7b_v2"

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    instruction = "Explain the following smart contract code:"
    prompt = '#!opshin\n\n\ndef validator(n: int) -> int:\n    # Tuple assignment works\n    a, b = 3, n\n    # control flow via if, for and while\n    if b < 5:\n        print("add")\n        a += 5\n    while b < 5:\n        b += 1\n    for i in range(2):\n        print("loop", i)\n\n    # sha256, sha3_256 and blake2b\n    from hashlib import sha256 as hsh\n\n    x = hsh(b"123").digest()\n\n    # bytestring slicing, assertions\n    assert x[1:3] == b"e" + b"\\xa4", "Hash is wrong"\n\n    # create lists, check their length, add up integers\n    y = [1, 2]\n    return a + len(x) + len(y) if y[0] == 1 else 0\n'
    model_input = format_prompt(instruction, prompt)
    # Generate response
    response = generate_response(model, tokenizer, model_input)
    print("Explanation:")
    print(response)


if __name__ == "__main__":
    main()
