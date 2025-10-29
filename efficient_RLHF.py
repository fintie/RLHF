# efficient_rlhf.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import Dataset
import numpy as np
from typing import Dict, List
import random
from tqdm import tqdm

class SyntheticLabeler:

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model for synthetic labeling: {e}")
            self.model = None
            self.tokenizer = None
        
    def generate_synthetic_feedback(self, prompts: List[str]) -> List[Dict]:

        if self.model is None:
            return self._generate_mock_feedback(prompts)
            
        synthetic_data = []
        
        for prompt in tqdm(prompts[:50], desc="Generating synthetic feedback"):  
            responses = self._generate_responses(prompt, num_responses=2)
            
            if len(responses) >= 2:
                chosen = max(responses, key=len)
                rejected = min(responses, key=len)
                
                synthetic_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "synthetic": True
                })
        
        return synthetic_data
    
    def _generate_responses(self, prompt: str, num_responses: int = 2) -> List[str]:

        if self.model is None:
            return [f"Response A to {prompt}", f"Response B to {prompt}"]
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        
        responses = []
        for _ in range(num_responses):
            try:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(prompt, "").strip()
                responses.append(response)
            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append(f"Generated response for {prompt[:20]}...")
                
        return responses
    
    def _generate_mock_feedback(self, prompts: List[str]) -> List[Dict]:

        return [
            {
                "prompt": prompt,
                "chosen": f"Good response to: {prompt}",
                "rejected": f"Bad response to: {prompt}",
                "synthetic": True
            }
            for prompt in prompts[:30]  # 限制数量
        ]

class EfficientRLHFTrainer:

    def __init__(self, base_model_name: str = "microsoft/DialoGPT-small"):
        self.base_model_name = base_model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading tokenizer: {e}")

            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.synthetic_labeler = SyntheticLabeler(base_model_name)
        self.phase1_data = []
        
    def phase1_synthetic_warm_start(self, prompts: List[str]) -> List[Dict]:

        print("Starting Phase 1: Synthetic Warm-Start")
        
        synthetic_data = self.synthetic_labeler.generate_synthetic_feedback(prompts)
        self.phase1_data = synthetic_data
        
        print(f"Generated {len(synthetic_data)} synthetic preference pairs")
        return synthetic_data
    
    def phase3_dpo_optimization(self, data: List[Dict]):

        print("Starting Phase 3: DPO Optimization")
        
        train_dataset = Dataset.from_list([
            {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }
            for example in data
        ])
        
        try:
            model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            
            training_args = TrainingArguments(
                output_dir="./efficient_rlhf_results",
                per_device_train_batch_size=2,
                num_train_epochs=1,
                learning_rate=1e-5,
                logging_steps=5,
                save_steps=50,
                remove_unused_columns=False,
            )
            
            dpo_trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
                beta=0.1,
                max_prompt_length=128,
                max_length=256,
            )
            
            print("Starting DPO training...")
            dpo_trainer.train()
            dpo_trainer.save_model("./efficient_rlhf_final_model")
            print("DPO optimization completed!")
            
            return dpo_trainer
        except Exception as e:
            print(f"Error during DPO optimization: {e}")
            return None

def run_efficient_rlhf_demo():

    print("Running EfficientRLHF Demo")
    

    trainer = EfficientRLHFTrainer()
    

    test_prompts = [
        "What is machine learning?",
        "Explain reinforcement learning",
        "How does RLHF work?",
        "What are the benefits of AI safety?",
        "Describe the transformer architecture",
    ] * 10 
    

    synthetic_data = trainer.phase1_synthetic_warm_start(test_prompts)
    

    final_trainer = trainer.phase3_dpo_optimization(synthetic_data)
    
    if final_trainer:
        print("EfficientRLHF demo completed successfully!")
    else:
        print("EfficientRLHF demo completed with some errors (expected in demo mode)")

if __name__ == "__main__":
    run_efficient_rlhf_demo()