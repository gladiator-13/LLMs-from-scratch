"""
Input Embeddings for a Transformer Language Model

This script:
1. Loads a text dataset
2. Tokenizes it using tiktoken
3. Creates input–target pairs
4. Generates token and positional embeddings
"""

try:
  with open("The_Time_Machine.txt", "r", encoding="utf-8") as file:
    data = file.read()
except FileNotFoundError:
  FileNotFoundError("The_Time_Machine.txt not found in the current directory.")

print(len(data))

"""Creating the input–target pairs as pytorch tensors using Dataset and Dataloader library of pytorch"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class LlmDataset(Dataset):
  """
  Creates input–target token sequences using a sliding window.
  Each input sequence predicts the next token sequence.
  """
  def __init__(self, txt, context_length, stride):
    self.input_ids = []
    self.target_ids = []

    #Tokenization step
    tokenizer = tiktoken.get_encoding('o200k_base')
    token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

    # Sliding window to group the sentences into overlapping sequences of context_length
    for i in range(0, len(token_ids)-context_length, stride):
      input_seq = token_ids[i:i+context_length]
      target_seq = token_ids[i+1:i+context_length+1]
      self.input_ids.append(torch.tensor(input_seq))
      self.target_ids.append(torch.tensor(target_seq))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, ids):
    return self.input_ids[ids], self.target_ids[ids]

def create_dataloader(txt, batch_size=4, context_length=256, drop_last=True, shuffle=True, stride=128, num_workers=0):
  dataset = LlmDataset(txt, context_length, stride)

  dataloader = DataLoader(
      dataset,
      shuffle=shuffle,
      drop_last = drop_last,
      batch_size = batch_size,
      num_workers = num_workers
  )

  return dataloader

"""# Token Embeddings

**Token Embeddings** :

A token embedding is a vector representation of a token (word, subword, or character). Since neural networks cannot understand text directly, every token must be converted into numbers. Instead of using simple IDs, we map each token to a dense vector in a high-dimensional space.

Creating token embeddings using PyTorch's Embeddings module
"""

import torch

# o200k_base vocabulary has a size of approx 200k
vocab_size = 200000
embedding_dim = 256

torch.manual_seed(123)
embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
#Creates a matrix of 200k x 256 dimension where each row repesents a token embedding

print(embeddings.weight)

"""Dataloader creates a matrix of tokenIDs where each row contains 64 tokens and the next row contains is made by sliding by 32 tokens."""

data_loader = create_dataloader(data, batch_size=8, context_length=64, stride=32, shuffle=False)

data_iter = iter(data_loader)
first_batch = next(data_iter)
print(first_batch)

"""Now each token will be replaced by it's token embeddings"""

token_embeddings = embeddings(first_batch[0])

print(token_embeddings.shape)
#The first batch contain 8 rows and 64 columns and each token is replaces with 256 dimension vector

"""# Positional Embeddings

**Positional Embeddings** :

Transformers do not have built-in sense of order. Unlike RNNs or CNNs, attention mechanisms process tokens in parallel.
So without positional information:

"The cat sat"
"Sat cat the"

would look identical to the model.

For each batch the positional encoding will remain the same for each row as we just want the positions of the values given as input to the transformers.
"""

context_length = 64
output_dim = 256

positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embedding = positional_embedding_layer(torch.arange(context_length))
print(pos_embedding.shape)

"""# Input Embeddings

Final input vector = Token embedding + Positional embedding
"""

input_embeddings = token_embeddings + pos_embedding
print(input_embeddings.shape)