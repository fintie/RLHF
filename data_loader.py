# data_loader.py
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List
import os

class RLHFDataLoader:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_hh_rlhf(self, split: str = "train", sample_size: int = None):

        try:
            dataset = load_dataset("Anthropic/hh-rlhf", split=split)
            if sample_size:
                dataset = dataset.select(range(min(sample_size, len(dataset))))
            return dataset
        except Exception as e:
            print(f"Error loading HH-RLHF: {e}")

            return self._create_mock_data(100 if sample_size is None else sample_size)
    
    def load_stack_exchange(self, split: str = "train", sample_size: int = None):

        try:
            dataset = load_dataset("lvwerra/stack-exchange-paired", split=split)
            if sample_size:
                dataset = dataset.select(range(min(sample_size, len(dataset))))
            return dataset
        except Exception as e:
            print(f"Error loading Stack Exchange: {e}")
            return self._create_mock_data(100 if sample_size is None else sample_size)
    
    def _create_mock_data(self, num_examples: int):

        from datasets import Dataset
        import random
        
        mock_data = []
        for i in range(num_examples):
            mock_data.append({
                "prompt": f"This is a test prompt {i}",
                "chosen": f"This is a good response to prompt {i}",
                "rejected": f"This is a bad response to prompt {i}"
            })
        
        return Dataset.from_list(mock_data)
    
    def preprocess_pair(self, example: Dict, dataset_type: str = "hh-rlhf") -> Dict:

        if dataset_type == "hh-rlhf" and "chosen" in example and "rejected" in example:
            chosen = example["chosen"]
            rejected = example["rejected"]

            if "\n\nHuman: " in chosen:
                chosen = chosen.split("\n\nAssistant: ")[-1]
            if "\n\nHuman: " in rejected:
                rejected = rejected.split("\n\nAssistant: ")[-1]
        else:
            chosen = example.get("chosen", "Default chosen response")
            rejected = example.get("rejected", "Default rejected response")
            
        return {
            "prompt": example.get("prompt", "Default prompt"),
            "chosen": chosen,
            "rejected": rejected
        }


if __name__ == "__main__":
    loader = RLHFDataLoader()
    

    print("Testing data loader...")
    test_data = loader.load_hh_rlhf("train[:10]")
    print(f"Loaded {len(test_data)} examples")
    

    processed = test_data.map(
        lambda x: loader.preprocess_pair(x, "hh-rlhf"),
        batched=False
    )
    print("Data preprocessing completed!")