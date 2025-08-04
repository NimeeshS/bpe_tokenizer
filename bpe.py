import json
from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.merges = [] 
        self.vocab = set(range(256))
        self.next_id = 256
        self.merge_ranks = {}
        self.id_to_pair = {}
        self.encoded_tokens = []

    def train(self, text, vocab_size=512):
        tokens = list(text.encode('utf-8'))
        self.next_id = 256
        while len(self.vocab) < vocab_size:
            pairs = self._find_common_pairs(tokens)
            if not pairs:
                break
            top_pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_id = self.next_id
            self.merges.append((top_pair, new_id))
            self.vocab.add(new_id)
            self.merge_ranks[top_pair] = len(self.merges) - 1
            self.id_to_pair[new_id] = top_pair
            self.next_id += 1
            tokens = self._merge_tokens(tokens, top_pair, new_id)

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while True:
            pairs = self._find_common_pairs(tokens)
            candidates = [
                (pair, self.merge_ranks[pair])
                for pair in pairs if pair in self.merge_ranks
            ]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda x: x[1])[0]
            new_id = dict(self.merges)[best_pair]
            tokens = self._merge_tokens(tokens, best_pair, new_id)
        
        self.encoded_tokens = tokens
        return tokens

    def decode(self, tokens):
        def decode_token(token):
            if token < 256:
                return [token]
            pair = self.id_to_pair.get(token)
            if pair:
                return decode_token(pair[0]) + decode_token(pair[1])
            return []
        flat = []
        for token in tokens:
            flat.extend(decode_token(token))
        return bytes(flat).decode('utf-8', errors='replace')

    def save(self, filename):
        merges_to_save = [
            {'pair': list(pair), 'new_id': new_id} for pair, new_id in self.merges
        ]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(merges_to_save, f)

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            merges_loaded = json.load(f)
        self.merges = []
        self.vocab = set(range(256))
        self.next_id = 256
        self.merge_ranks = {}
        self.id_to_pair = {}
        for idx, item in enumerate(merges_loaded):
            pair = tuple(item['pair'])
            new_id = item['new_id']
            self.merges.append((pair, new_id))
            self.vocab.add(new_id)
            self.merge_ranks[pair] = idx
            self.id_to_pair[new_id] = pair
            self.next_id = max(self.next_id, new_id + 1)

    def _find_common_pairs(self, tokens):
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] += 1
        return pairs

    def _merge_tokens(self, tokens, pair, new_id):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def compression_rate(self, text):
        original_bytes = len(text.encode('utf-8'))
        compressed_tokens = len(self.encoded_tokens)
        ratio = original_bytes / compressed_tokens if compressed_tokens else 0
        print(f"Original bytes: {original_bytes}")
        print(f"Compressed tokens: {compressed_tokens}")
        print(f"Compression ratio (bytes per token): {ratio:.2f}")
        return ratio
    
    def visualize_tokens(self, text=None):
        """
        Visualize how text splits into tokens after encoding.
        Shows each token as the characters it represents.
        """
        if self.encoded_tokens == []:
            raise ValueError("No tokens encoded yet. Please encode text first.")
        if text is None:
            tokens = self.encoded_tokens
        else:
            tokens = self.encode(tokens)
        parts = []
        for token in tokens:
            decoded_bytes = self._decode_token_to_bytes(token)
            try:
                decoded_str = bytes(decoded_bytes).decode('utf-8')
            except UnicodeDecodeError:
                decoded_str = "�"
            parts.append(f"[{decoded_str}]")
        print(''.join(parts))
        return parts

    def _decode_token_to_bytes(self, token):
        """
        Recursively decode a token ID to the list of bytes it represents.
        """
        if token < 256:
            return [token]
        pair = self.id_to_pair.get(token)
        if pair:
            return self._decode_token_to_bytes(pair[0]) + self._decode_token_to_bytes(pair[1])
        return []

    def draw_merge_tree(self, token, prefix=''):
        """
        Recursively print how a token was built from its sub-tokens.
        """
        if token < 256:
            ch = chr(token) if 32 <= token <= 126 else f'\\x{token:02x}'
            print(f"{prefix}'{ch}'")
        else:
            print(f"{prefix}{token}")
            pair = self.id_to_pair.get(token)
            if pair:
                left, right = pair
                self.draw_merge_tree(left, prefix + "├── ")
                self.draw_merge_tree(right, prefix + "└── ")

    def visualize_merge_trees(self, text=None):
        """
        Encode text and draw merge trees for each resulting token.
        """
        if self.encoded_tokens == []:
            raise ValueError("No tokens encoded yet. Please encode text first.")
        if text is None:
            tokens = self.encoded_tokens
        else:
            tokens = self.encode(text)
        for token in tokens:
            print(f"\nTree for token {token}:")
            self.draw_merge_tree(token)

    def list_tokens(self):
        for pair, new_id in self.merges:
            try:
                decoded_str = bytes(self._decode_token_to_bytes(new_id)).decode('utf-8')
            except UnicodeDecodeError:
                decoded_str = "�"
            print(f'"{decoded_str}"')