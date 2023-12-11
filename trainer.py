
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler

import os
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

class DistributedRLHFTrainer:
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        set_seed(self.config.seed)
        setup_distributed(self.rank, self.world_size)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pre-trained model for PPO
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])

        # Load reward model
        self.reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name)
        self.reward_model.to(rank)
        self.reward_model.eval() # Reward model should be in eval mode

        self.ppo_trainer = PPOTrainer(
            config=PPOConfig(**config.ppo_config),
            model=self.model,
            ref_model=None, # In a real scenario, this would be a frozen copy of the initial model
            tokenizer=self.tokenizer,
            dataset=self._get_dummy_dataset(), # Replace with actual dataset
        )

        self.response_length_sampler = LengthSampler(config.min_response_length, config.max_response_length)

    def _get_dummy_dataset(self):
        # In a real application, load your actual dataset here
        # For demonstration, we create a dummy dataset
        dummy_data = [
            {"query": "Explain machine learning.", "input_ids": self.tokenizer.encode("Explain machine learning.", return_tensors="pt")[0]},
            {"query": "What is reinforcement learning?", "input_ids": self.tokenizer.encode("What is reinforcement learning?", return_tensors="pt")[0]},
        ]
        return dummy_data

    def _generate_response(self, query_tensor):
        gen_len = self.response_length_sampler()
        generation_kwargs = {
            "min_length": gen_len,
            "max_length": gen_len,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        response = self.ppo_trainer.generate(query_tensor, **generation_kwargs)
        return response

    def _get_reward(self, texts):
        # Use the reward model to score responses
        inputs = self.reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.rank)
        with torch.no_grad():
            rewards = self.reward_model(**inputs).logits.squeeze(-1)
        return rewards

    def train(self, num_epochs=1):
        for epoch in range(num_epochs):
            for batch in self.ppo_trainer.dataloader:
                query_tensors = batch["input_ids"]
                
                # Generate responses
                response_tensors = self._generate_response(query_tensors)
                response_texts = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

                # Get rewards
                rewards = self._get_reward(response_texts)

                # PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards)

                if self.rank == 0:
                    print(f"Epoch {epoch}, Stats: {stats}")

    def save_model(self, path="./rlhf_model"):
        if self.rank == 0:
            self.ppo_trainer.save_pretrained(path)
            print(f"Model saved to {path}")

    def __del__(self):
        cleanup_distributed()

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TrainingConfig:
        model_name: str = "gpt2"
        reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large"
        seed: int = 42
        min_response_length: int = 16
        max_response_length: int = 64
        ppo_config: dict = {
            "learning_rate": 1e-5,
            "batch_size": 1,
            "mini_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "total_steps": 20000,
            "log_with": "tensorboard",
            "accelerator_kwargs": {"num_processes": 1},
        }

    # This example assumes a single GPU setup for simplicity.
    # For multi-GPU, you would typically use torch.multiprocessing.spawn
    # to launch multiple processes, each running this script with a different rank.
    world_size = 1 # Number of GPUs/processes
    rank = 0     # Current process rank

    config = TrainingConfig()
    trainer = DistributedRLHFTrainer(config, rank, world_size)
    trainer.train(num_epochs=1)
    trainer.save_model()

# Commit timestamp: 2023-12-11 00:00:00 - 281
