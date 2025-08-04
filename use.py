from bpe import BPETokenizer

def main():
    with open('source.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.load('bpe_merges.json')
    tokens = tokenizer.encode(text)
    print("Encoded tokens:", tokens[:10])

    decoded = tokenizer.decode(tokens)
    print("Decoded text matches original:", decoded == text)

    tokenizer.compression_rate(text)
    tokenizer.visualize_tokens()
    tokenizer.visualize_merge_trees()
    tokenizer.list_tokens()

if __name__ == "__main__":
    main()