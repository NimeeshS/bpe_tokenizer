# Byte Pair Encoding (BPE) Tokenizer

This project lets you:
- Train a BPE tokenizer on your own text data
- Encode new text into token IDs
- Decode token IDs back into text
- **Receive data** on how effective the tokenization was
- **Visualize** how text splits into tokensHo 
- **Draw merge trees** to see exactly how each token was built from sub-tokens
---
## **How to use it**

- Download all files 
- Add all training text data to source.txt
- Run train.py
- Once the token data is stored in bpe_merges.json, the text in source.txt can be deleted and replaced with any new text to tokenize
- Run use.py to see what the different token data visualization methods do