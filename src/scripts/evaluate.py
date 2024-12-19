import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import math


def format_prompt(instruction: str, input_str: str) -> str:
    if input_str.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def compute_perplexity(prompt: str, target: str, model, tokenizer) -> float:
    if not target.strip():
        return None
    full_sequence = prompt + target
    encoded = tokenizer(full_sequence, return_tensors="pt")
    input_ids_full = encoded.input_ids.to(model.device)
    prompt_ids = tokenizer(prompt).input_ids
    prompt_length = len(prompt_ids)
    labels = input_ids_full.clone()
    labels[:, :prompt_length] = -100
    with torch.no_grad():
        outputs = model(input_ids=input_ids_full, labels=labels)
        loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: Loss is NaN or Inf for this example.")
        return None
    loss_val = loss.item()
    if loss_val > 100:
        print("Warning: Loss extremely large, returning None for perplexity.")
        return None
    perplexity = math.exp(loss_val)
    return perplexity


def evaluate(
    test_dataset_path: str,
    output_path: str,
    tokenizer: LlamaTokenizer,
    model: LlamaForCausalLM,
):
    with open(test_dataset_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for i, line in enumerate(infile):
            print("Processing example", i)
            data = json.loads(line)
            instruction = data.get("instruction", "")
            input_str = data.get("input", "")
            output = data.get("output", "")
            file_path = data.get("file_path", "")
            file = data.get("file", "")
            prompt = format_prompt(instruction, input_str)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=1500,
                temperature=0.2,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            generated_tokens = generation_output[0][input_ids.size(-1) :]
            generated_answer = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            perplexity = compute_perplexity(prompt, output, model, tokenizer)
            print("Perplexity:", perplexity)
            print("Generated answer:", generated_answer)
            result = {
                "instruction": instruction,
                "input": input_str,
                "target": output,
                "output": generated_answer,
                "perplexity": perplexity,
                "file_path": file_path,
                "file": file,
            }
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    # Paths
    model_path = "unboundedmarket/smart_contract_explainer_open_llama_7b_v2"
    test_dataset_path = "src/data/test_dataset_postprocessed.jsonl"
    output_path = "src/data/predictions.jsonl"

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    evaluate(test_dataset_path, output_path, tokenizer, model)


if __name__ == "__main__":
    main()
