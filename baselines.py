# baselines.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import Dataset
import numpy as np
from typing import Dict

class BaselineReproducer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def setup_dpo_training(self, train_dataset: Dataset, eval_dataset: Dataset = None):

        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            training_args = TrainingArguments(
                output_dir="./dpo_results",
                per_device_train_batch_size=2,  
                per_device_eval_batch_size=2,
                num_train_epochs=1,  
                logging_steps=5,
                save_steps=100,
                evaluation_strategy="no",  
                learning_rate=1e-5,
                warmup_steps=10,
                logging_dir="./logs",
                remove_unused_columns=False,
                dataloader_drop_last=True,
            )
            
            dpo_trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                beta=0.1,
                max_prompt_length=128,  
                max_length=256,
            )
            
            return dpo_trainer
        except Exception as e:
            print(f"Error setting up DPO training: {e}")
            return None
    
    def train_dpo_baseline(self, train_dataset: Dataset, num_epochs: int = 1):

        print("Training DPO baseline...")
        
        dpo_trainer = self.setup_dpo_training(train_dataset)
        if dpo_trainer is None:
            print("Failed to setup DPO trainer")
            return None
            
        try:
            dpo_trainer.train()
            dpo_trainer.save_model("./dpo_baseline_model")
            print("DPO baseline training completed!")
            return dpo_trainer
        except Exception as e:
            print(f"Error during DPO training: {e}")
            return None


if __name__ == "__main__":
    from data_loader import RLHFDataLoader
    
    print("Testing baseline reproduction...")
    

    loader = RLHFDataLoader()
    test_data = loader.load_hh_rlhf("train[:20]")  
    

    processed_data = test_data.map(
        lambda x: loader.preprocess_pair(x, "hh-rlhf"),
        batched=False
    )
    

    dpo_data = Dataset.from_list([
        {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
        for example in processed_data
    ])
    

    reproducer = BaselineReproducer()
    trainer = reproducer.train_dpo_baseline(dpo_data)
    
    if trainer:
        print("Baseline reproduction test passed!")
    else:
        print("Baseline reproduction test failed, but this might be expected with small test data")