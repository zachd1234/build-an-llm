import os
from typing import List, Dict

class Tokenizer:
    def __init__(self, text: str = None, file_path: str = None):
        """
        Initialize the tokenizer with either text directly or from a file.
        
        Args:
            text (str, optional): Direct text input
            file_path (str, optional): Path to input file
        """
        if text is None and file_path is None:
            raise ValueError("Either text or file_path must be provided")
            
        if text is None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s: str) -> List[int]:
        """Encode a string into a list of integers."""
        return [self.stoi[c] for c in s]
    
    def decode(self, l: List[int]) -> str:
        """Decode a list of integers back into a string."""
        return ''.join([self.itos[i] for i in l])
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self) -> List[str]:
        """Return the vocabulary as a list of characters."""
        return self.chars
    
    def save_vocab(self, file_path: str):
        """Save the vocabulary to a file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for ch in self.chars:
                f.write(f"{ch}\n")
    
    @classmethod
    def load_vocab(cls, file_path: str) -> 'Tokenizer':
        """Load a tokenizer from a saved vocabulary file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f if line.strip()]
        text = ''.join(chars)  # Create dummy text
        return cls(text=text)

def main():
    # Example usage
    try:
        # Initialize tokenizer from file
        tokenizer = Tokenizer(file_path='input.txt')
        
        # Print vocabulary information
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        print(f"First 10 characters in vocabulary: {tokenizer.get_vocab()[:10]}")
        
        # Test encoding/decoding with text that exists in our input
        sample = "Speak, speak."
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        
        print("\nTesting encoding/decoding:")
        print("Sample input: ", sample)
        print("Encoded:      ", encoded)
        print("Decoded:      ", decoded)
        
        # Save vocabulary
        tokenizer.save_vocab('vocab.txt')
        print("\nVocabulary saved to vocab.txt")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 