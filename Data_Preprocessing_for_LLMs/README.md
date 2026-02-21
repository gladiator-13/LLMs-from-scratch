# Data Preprocessing for Transformer Language Model

This folder contains the data preprocessing pipeline used before training the Transformer model.

The preprocessing stage converts raw text into numerical representations that can be consumed by a neural network.

It includes:

1. Tokenization (text → token IDs)
2. Embedding generation (token IDs → vectors)

---

## 📂 Folder Structure

Data_Preprocessing/
│
├── tokenizer/
│   ├── simple_tokenizer.py
│   └── The_Gift_of_the_Magi.txt
│
├── embeddings/
│   ├── input_embeddings.py
│   └── The_Time_Machine.txt
│
└── README.md

---

## 🔹 1. Tokenizer Module

Location: tokenizer/simple_tokenizer.py

This module demonstrates:

- Regex-based word tokenization
- Vocabulary construction
- Mapping tokens ↔ token IDs
- Handling unknown tokens (`<|unk|>`)
- Use of special token (`<|endoftext|>`)
- Comparison with `tiktoken` (BPE)

### Implemented Tokenization Approaches

- Word-based tokenization (educational implementation)
- Byte Pair Encoding using `tiktoken` (`o200k_base`)

### Why This Matters

Neural networks cannot process raw text.  
Tokenization converts text into discrete token IDs that serve as input to embedding layers.

---

## 🔹 2. Embeddings Module

Location: embeddings/input_embeddings.py


This module implements:

- Sliding window dataset creation
- Input–target sequence generation
- Token embeddings (`nn.Embedding`)
- Positional embeddings
- Final input embedding computation

### Embedding Formula

Final input to transformer: Input Embedding = Token Embedding + Positional Embedding


Where:

- Token embedding captures semantic meaning
- Positional embedding encodes sequence order

---

## 🔁 Data Flow Overview

Raw Text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Embedding Layer
   ↓
Input Embeddings
   ↓
Transformer

---

## ⚙️ Design Decisions

- `o200k_base` tokenizer used for compatibility with modern GPT models
- Context length and stride implemented using sliding window
- Embeddings are initialized randomly and trained end-to-end
- Dataset built using PyTorch `Dataset` and `DataLoader`

---
