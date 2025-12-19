import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SupervisedDataset(Dataset):
    def __init__(self, vl_chat_processor, seq_len=512, total_samples=100000, 
                 min_num=-100, max_num=100, interval=0.01):
        self.tokenizer = vl_chat_processor.tokenizer
        self.seq_len = seq_len
        self.total_samples = total_samples
        self.all_num = np.linspace(min_num, max_num, int((max_num - min_num) / interval) + 1)
        self.min_num = min_num
        self.max_num = max_num
        self.interval = interval

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        sampled_nums = random.sample(list(self.all_num), self.seq_len)
        sampled_tokens = [f"{x:.2f}" for x in sampled_nums]
        token_ids = self.tokenizer.encode(" ".join(sampled_tokens), add_special_tokens=False)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        return {"input_ids": token_ids}
