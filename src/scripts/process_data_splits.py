import json
import random
from collections import defaultdict


def preprocess_and_split(input_file, train_file, test_file, test_ratio=0.1):
    """
    Preprocess the dataset and split it into training and test sets.
    """
    data_by_type = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            file_type = record.get("processed_file", "").split(".")[-1]
            data_by_type[file_type].append(record)

    train_data = []
    test_data = []

    for file_type, records in data_by_type.items():
        random.shuffle(records)
        test_size = int(len(records) * test_ratio)
        test_split = records[:test_size]
        train_split = records[test_size:]

        for record in test_split:
            test_data.append(
                {
                    "instruction": "Explain the following smart contract code:",
                    "input": record.get("contract", ""),
                    "output": record.get("explanation", ""),
                    "file_path": record.get("file_path", ""),
                    "file": record.get("file", ""),
                }
            )

        for record in train_split:
            train_data.append(
                {
                    "instruction": "Explain the following smart contract code:",
                    "input": record.get("contract", ""),
                    "output": record.get("explanation", ""),
                }
            )

    with open(train_file, "w", encoding="utf-8") as train_out, open(
        test_file, "w", encoding="utf-8"
    ) as test_out:
        for record in train_data:
            train_out.write(json.dumps(record) + "\n")
        for record in test_data:
            test_out.write(json.dumps(record) + "\n")

    print(
        f"Preprocessing complete. Training set: {len(train_data)} examples, Test set: {len(test_data)} examples."
    )


def main():
    input_file = "src/data/processed/contracts_dataset.jsonl"
    train_file = "src/data/processed/train_dataset.jsonl"
    test_file = "src/data/processed/test_dataset.jsonl"
    preprocess_and_split(input_file, train_file, test_file)


if __name__ == "__main__":
    main()
