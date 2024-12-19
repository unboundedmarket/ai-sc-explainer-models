import json
import tiktoken


def load_jsonl(file_path):
    """Load a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def estimate_tokens(data, encoding_name="cl100k_base"):
    """Estimate token counts for each entry in the dataset."""
    encoding = tiktoken.get_encoding(encoding_name)
    token_stats = []
    for entry in data:
        input_instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")
        input_tokens = len(encoding.encode(input_instruction + input_text))
        output_tokens = len(encoding.encode(output_text))
        total_tokens = input_tokens + output_tokens
        token_stats.append(
            {
                "entry": entry,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )
    return token_stats


def display_summary(token_stats, target=2048):
    """Display a summary of token statistics."""
    total_inputs = sum(item["input_tokens"] for item in token_stats)
    total_outputs = sum(item["output_tokens"] for item in token_stats)
    total_combined = sum(item["total_tokens"] for item in token_stats)
    avg_input = total_inputs / len(token_stats)
    avg_output = total_outputs / len(token_stats)
    avg_combined = total_combined / len(token_stats)
    max_input = max(item["input_tokens"] for item in token_stats)
    max_output = max(item["output_tokens"] for item in token_stats)
    max_combined = max(item["total_tokens"] for item in token_stats)
    above_target = len([item for item in token_stats if item["input_tokens"] > target])
    print(f"Total examples: {len(token_stats)}")
    print(f"Average input tokens: {avg_input:.2f}")
    print(f"Average output tokens: {avg_output:.2f}")
    print(f"Average combined tokens: {avg_combined:.2f}")
    print(f"Maximum input tokens: {max_input}")
    print(f"Maximum output tokens: {max_output}")
    print(f"Maximum combined tokens: {max_combined}")
    print(f"Total input tokens: {total_inputs}")
    print(f"Total output tokens: {total_outputs}")
    print(f"Total combined tokens: {total_combined}")
    print(f"Examples above {target} tokens: {above_target}")


def postprocess_data(token_stats, target=2048):
    """Remove entries with total tokens above the target threshold."""
    filtered_data = [
        item["entry"] for item in token_stats if item["input_tokens"] <= target
    ]
    return filtered_data


def main():
    file_path = "src/data/test_dataset.jsonl"
    output_path = "src/data/test_dataset_processed.jsonl"
    target_token_limit = 1024
    data = load_jsonl(file_path)
    token_stats = estimate_tokens(data)
    display_summary(token_stats, target=target_token_limit)
    filtered_data = postprocess_data(token_stats, target=target_token_limit)
    save_jsonl(filtered_data, output_path)
    print(f"Postprocessed data saved to {output_path}")


if __name__ == "__main__":
    main()
