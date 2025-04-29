# Build an LLM 🧠

A lightweight, educational large language model (LLM) built from scratch using PyTorch — featuring custom tokenization, embeddings, and transformer-based self-attention.

## 🛠️ Tech Stack
- Python 3
- PyTorch (torch)
- Virtualenv (venv)

## 🧩 Main Components

### 🔤 Tokenizer
- Custom character-level tokenizer
- Vocabulary building, encoding, decoding
- Vocabulary persistence (save/load `vocab.txt`)
- UTF-8 compatible
- Supports file-based and direct text input

### 🔗 Embeddings
- Word embeddings implemented with `torch.nn.Embedding`
- Configurable embedding dimensions (default: 32)
- Seamless integration with tokenizer

### 🧠 Transformer
- Simplified transformer architecture
- Core components:
  - Self-attention mechanism (Query, Key, Value projections)
  - Scaled dot-product attention
  - Causal masking (`torch.tril`) to enforce autoregression
  - Layer normalization for training stability
- Configurable parameters:
  - Embedding dimension: 32
  - Head size: 16
  - Block size (sequence length): 16

## 📚 Project Purpose
This project is designed as an educational exploration of LLM internals.  
Rather than using heavy frameworks, it **builds the core logic manually** — helping internalize how tokenization, embedding spaces, attention heads, and sequence modeling work at a fundamental level.

Focused on:
- Clarity
- Modularity
- Core transformer mechanics without external abstractions

## 🗃️ Project Structure
