import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import Tokenizer  # We'll use our existing tokenizer

class Head(nn.Module):
    def __init__(self, head_size, embed_dim, block_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.ln = nn.LayerNorm(embed_dim)  # Add layer normalization
        self.block_size = block_size

    def forward(self, x):
        B, T, C = x.shape
        
        # Ensure sequence length doesn't exceed block_size
        if T > self.block_size:
            x = x[:, :self.block_size, :]
            T = self.block_size
        
        # Apply layer norm before attention
        x = self.ln(x)
        
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # Attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # normalized

        v = self.value(x)   # (B, T, head_size)
        out = wei @ v       # (B, T, head_size)
        return out

def main():
    # Initialize tokenizer
    tokenizer = Tokenizer(file_path='input.txt')
    
    # Parameters
    embed_dim = 32
    head_size = 16
    block_size = 16  # Increased block size to handle longer sequences
    
    # Create embedding table
    embedding_table = nn.Embedding(tokenizer.get_vocab_size(), embed_dim)
    
    # Create attention head
    head = Head(head_size, embed_dim, block_size)
    
    # Test with a sample text
    sample_text = "Speak, speak."
    print(f"\nSample text: {sample_text}")
    
    # Get embeddings
    x = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)
    x = embedding_table(x)  # (T, embed_dim)
    x = x.unsqueeze(0)     # Add batch dimension (1, T, embed_dim)
    
    # Apply attention
    out = head(x)
    
    # Print shapes
    print("\nShapes:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Print attention weights for first token
    print("\nFirst token attention weights:")
    print(head(x)[0, 0, :5], "...")  # First 5 dimensions of first token

if __name__ == "__main__":
    main() 