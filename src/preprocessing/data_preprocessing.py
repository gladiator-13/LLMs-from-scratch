import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class LlmDataset(Dataset):
  def __init__(self, txt, context_length, stride):
    self.input_ids = []
    self.target_ids = []

  # Tokenization
  tokenizer = tiktoken.get_encoding('o200k_base')
  token_ids = tokenizer.encode(txt, allowed_special={'|<endoftext>|'})

  # Sliding window to group the sentences into overlapping sequences of context_length
  for i in range(0, len(token_ids)-context_length, stride):
    input_seq = token_ids[i:i+context_length]
    target_seq = token_ids[i+1:i+context_length+1]

    self.input_ids.append(input_seq)
    self.target_ids.append(target_seq)

  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, ids):
    return self.input_ids[ids], self.target_ids[ids]
  

def create_dataloader(txt, batch_size=8, context_length=256, drop_last=True, shuffle=True, stride=128, num_workers=0):
  dataset = LlmDataset(txt, context_length, stride)

  dataloader = DataLoader(
      dataset,
      shuffle=shuffle,
      drop_last = drop_last,
      batch_size = batch_size,
      num_workers = num_workers
  )

  return dataloader

