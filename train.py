from bpe import BPETokenizer

def main():
    with open('source.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=512)
    tokenizer.save('bpe_merges.json')
    print("Tokenizer trained and saved.")

if __name__ == "__main__":
    main()