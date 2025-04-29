import torch
import torch.nn as nn
from tokenizer import Tokenizer

def main():
    # Initialize tokenizer
    tokenizer = Tokenizer(file_path='input.txt')
    
    # Embedding parameters
    vocab_size = tokenizer.get_vocab_size()
    embed_dim = 32  # You can adjust this
    
    # Create embedding table
    embedding_table = nn.Embedding(vocab_size, embed_dim)
    
    # Test with a sample text
    sample_text = "Speak, speak."  # Using text we know exists in our vocabulary
    print(f"\nSample text: {sample_text}")
    
    # Encode input text
    x = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)  # Shape: (T,)
    print("Token IDs:", x)
    
    # Look up embeddings
    embeddings = embedding_table(x)  # Shape: (T, embed_dim)
    print("Embedding shape:", embeddings.shape)
    print("First embedding vector:", embeddings[0][:5], "...")  # Show first 5 dimensions of first embedding
    
    # Print some statistics
    print("\nEmbedding Statistics:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of tokens in sample: {len(x)}")
    print(f"Total embedding parameters: {vocab_size * embed_dim}")

if __name__ == "__main__":
    main() 