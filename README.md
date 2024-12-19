# AI Explainer Models

Welcome to the **AI Explainer Models**!  
This repository hosts our language models fine-tuned to explain Cardano smart contracts.

For details about the interface for visualizing and browsing the contracts, visit:  
[**AI Smart Contract Explainer Interface**](https://github.com/unboundedmarket/ai-sc-explainer-interface).  

## Overview

The first version (V1) of our smart contract explainer model is available on Hugging Face:  
[`unboundedmarket/smart_contract_explainer_open_llama_7b_v2`](https://huggingface.co/unboundedmarket/smart_contract_explainer_open_llama_7b_v2)

This model is based on the fine-tuned **Open Llama 7B V2**.  
You can find the base model here:  
[Open Llama 7B V2](https://huggingface.co/openlm-research/open_llama_7b_v2)

The fine-tuning was performed on a dateset of Cardano smart contracts using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) on an NVIDIA L40 GPU.

---

## Getting Started

### Clone the Repository

Clone this repository:

```bash
git clone https://github.com/unboundedmarket/ai-sc-explainer-models.git
cd ai-sc-explainer-models
```

### Install Dependencies

Ensure you have all dependencies installed by running:

```bash
pip install -r requirements.txt
```

---

## Model Usage

### Load the Model

You can directly load the model using the following code snippet:

```python
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = "unboundedmarket/smart_contract_explainer_open_llama_7b_v2"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
```

### Example Inference

For an example of how to generate explanations for Cardano smart contracts, see:  
`src/script/inference.py`

---

## Training Data

The model was fine-tuned using a dataset of Cardano smart contracts. The training data is available in:  
`src/data/train_dataset.jsonl`
