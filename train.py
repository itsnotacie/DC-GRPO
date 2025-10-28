from trl import GRPOConfig, GRPOTrainer
from dcgrpo import DCGRPOTrainer
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from cirron import Collector
from testing_util import run_test
from sklearn import preprocessing
import numpy as np
import sys

sys.setrecursionlimit(2**31-1)

model_names = {
    "Qwen": "Qwen/Qwen2.5-Coder-1.5B-Instruct", 
    "DeepSeek": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "Llama": "meta-llama/Llama-3.2-1B-Instruct",
}

def extract_code(completions):
    code_block_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    code_block_match = code_block_pattern.search(completions)
    code = code_block_match.group(1).strip() if code_block_match else completions
    return code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the model to use for code generation.', required=True, type=str)
    parser.add_argument('--task', help='the specific task the model will perform.', required=True, type=str)

    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(model_names[args.model], torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_names[args.model])
    output_dir = f"./saved_models/{args.task}/{args.model}-DCGRPO"
    dataset = load_from_disk(f"./grpo_data")
    
    def get_reward(efficiencies, pass_rewards):
        rewards = []
        scaler = preprocessing.MinMaxScaler()
        valid_data = [x for x in efficiencies if x is not None]
        if not valid_data:
            return pass_rewards
        max_eff = max(valid_data)
        for i in range(len(efficiencies)):
            if efficiencies[i] is None:
                efficiencies[i] = max_eff
        efficiencies = scaler.fit_transform(np.array(efficiencies).reshape(-1, 1))
        print(efficiencies)
        for i in range(len(efficiencies)):
            print(1-efficiencies[i][0]+pass_rewards[i])
            rewards.append(1-efficiencies[i][0]+pass_rewards[i])
        return rewards

    def efficiency_reward(completions, **kwargs) -> list[float]:
        rewards = []
        pass_rewards = []
        efficiencies = []
        for idx, completion in enumerate(completions):
            try:
                with Collector() as collector:
                    results = run_test(kwargs["task_id"][idx], extract_code(completion[0]["content"]))
                pass_reward = sum(1 for result in results if result is True) / len(results)
                pass_rewards.append(pass_reward)
            except Exception as e:
                print(f"Error processing completion {idx}: {e}")
                pass_rewards.append(0)
        rewards = get_reward(efficiencies, pass_rewards)
        print(rewards)
            
        return rewards


    training_args = GRPOConfig(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=8,
        num_train_epochs=2,
        logging_steps=1,
        max_completion_length=512,
        save_strategy="epoch",
        vllm_device="auto",
        vllm_gpu_memory_utilization=0.9,
        num_generations=8,
    )

    trainer = DCGRPOTrainer(
        model=model,
        reward_funcs=[
            efficiency_reward
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)