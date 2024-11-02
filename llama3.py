import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import random
from tqdm import tqdm
import os

def extract_ans(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

class LlamaEvaluator: 
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"Loading from {os.path.abspath(model_path)}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def n_shot_encode(self, data, question, n=8):
        chats = []
        for sample in random.sample(data, n):
            chats.append({"role": "user", 
                          "content": f"Question: {sample['question']}"})
            chats.append({"role": "assistant", 
                          "content": f"Answer: {sample['answer']}"})
        chats.append({"role": "user", 
                      "content": f"Question: {question} Let's think step by step. Write your anwser as an integer after ####"})
        return chats

    def evaluate_benchmark(self, benchmark_data):
        correct = 0
        max_length = 5000
        
        for i, example in enumerate(tqdm(benchmark_data)):
            prompt = self.n_shot_encode(benchmark_data, example)
            inputs = self.tokenizer.apply_chat_template(prompt,
                                                        return_tensors="pt",
                                                        padding=True,
                                                        truncation=True).to(self.device)
            all_logits = []
            generated_tokens = []
            entropy = 0
            
            with torch.no_grad():
                for num_generated in range(max_length):
                    outputs = self.model(
                        input_ids=inputs,
                    )
                    logits = outputs.logits[:, -1, :].to(torch.float32)
                    probs = torch.nn.functional.softmax(logits)
                    entropy += torch.nn.functional.cross_entropy(probs, probs).cpu().item()
                    
                    next_token = torch.argmax(logits, dim=-1)
                    generated_tokens.append(next_token.item())
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)

            generated_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) 
            reference_answer = example["answer"]
            if extract_ans(generated_answer) == extract_ans(reference_answer):
                correct+=1
            
            if (i+1) % 10 == 0:
                print(f"\nCurrent accuracy: {correct/(i+1):.2%} ({correct}/{i+1})")
        
        return correct/len(benchmark_data)


def main():
    model_path = "./llama3-instruct/"
    evaluator = LlamaEvaluator(model_path)
    
    print("Loading dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    dataset = [
        {"question": example["question"], "answer": example["answer"]}
        for example in dataset
    ]
    
    print("Starting evaluation...")
    results = evaluator.evaluate_benchmark(dataset)
    
    print(f"Overall accuracy: {results:.2%}")

if __name__ == "__main__":
    main()
