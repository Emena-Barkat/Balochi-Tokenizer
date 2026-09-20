"""
Balochi BPE Tokenizer
Trains a Byte-Pair Encoding (BPE) subword tokenizer on Balochi text dataset
and evaluates reconstruction accuracy and fertility on test dataset.
"""

import os
import json
import collections
from typing import List, Dict, Tuple, Optional

class BalochiBPETokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    MASK_TOKEN = "<mask>"

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.merge_list: List[Tuple[str, str]] = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        specials = [self.BOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN, self.MASK_TOKEN]
        self.token_to_id = {token: i for i, token in enumerate(specials)}
        self.id_to_token = {i: token for i, token in enumerate(specials)}
        self.merges = {}
        self.merge_list = []

    def train(self, corpus_path: str, vocab_size: int = 2000, min_frequency: int = 2):
        print(f"==> Training BPE Tokenizer on: {corpus_path}")
        self._init_special_tokens()

        word_freqs = collections.Counter()
        alphabet = set()

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word_freqs[line] += 1
                alphabet.update(line)

        print(f"--> Unique entries: {len(word_freqs)}, Base characters: {len(alphabet)}")

        # Add base alphabet
        for char in sorted(alphabet):
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char

        splits = {word: list(word) for word in word_freqs}

        while len(self.token_to_id) < vocab_size:
            pair_counts = collections.Counter()
            for word, freq in word_freqs.items():
                w_tokens = splits[word]
                for i in range(len(w_tokens) - 1):
                    pair_counts[(w_tokens[i], w_tokens[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_frequency:
                break

            rank = len(self.merges)
            self.merges[best_pair] = rank
            self.merge_list.append(best_pair)

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[merged_token] = new_id
                self.id_to_token[new_id] = merged_token

            # Update splits
            for word in word_freqs:
                w_tokens = splits[word]
                if len(w_tokens) < 2:
                    continue
                new_tokens = []
                i = 0
                while i < len(w_tokens):
                    if i < len(w_tokens) - 1 and (w_tokens[i], w_tokens[i + 1]) == best_pair:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(w_tokens[i])
                        i += 1
                splits[word] = new_tokens

        print(f"==> Training finished. Vocab size: {len(self.token_to_id)}, Merges: {len(self.merges)}")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = list(text)
        if len(tokens) < 2 or not self.merges:
            return tokens

        while True:
            min_rank = float("inf")
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges and self.merges[pair] < min_rank:
                    min_rank = self.merges[pair]
                    best_pair = pair

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            if len(tokens) < 2:
                break

        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        unk_id = self.token_to_id[self.UNK_TOKEN]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        specials = {self.BOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN, self.MASK_TOKEN}
        for tid in token_ids:
            tok = self.id_to_token.get(tid, self.UNK_TOKEN)
            if tok not in specials:
                tokens.append(tok)
        return "".join(tokens)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair in self.merge_list:
                f.write(f"{pair[0]} {pair[1]}\n")

        config = {
            "model_type": "bpe",
            "language": "balochi",
            "vocab_size": len(self.token_to_id),
            "num_merges": len(self.merge_list),
            "bos_token": self.BOS_TOKEN,
            "eos_token": self.EOS_TOKEN,
            "unk_token": self.UNK_TOKEN,
            "pad_token": self.PAD_TOKEN,
            "mask_token": self.MASK_TOKEN,
        }
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, model_dir: str):
        with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.merges = {}
        self.merge_list = []
        with open(os.path.join(model_dir, "merges.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            rank = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    pair = (parts[0], parts[1])
                    self.merges[pair] = rank
                    self.merge_list.append(pair)
                    rank += 1

    def evaluate(self, test_path: str):
        print("\n========================================================")
        print(f"==> Evaluating on Test Dataset: {test_path}")
        print("========================================================")

        with open(test_path, "r", encoding="utf-8") as f:
            test_lines = [l.strip() for l in f if l.strip()]

        total_chars = sum(len(l) for l in test_lines)
        total_tokens = 0
        exact_matches = 0
        unk_count = 0
        unk_id = self.token_to_id[self.UNK_TOKEN]

        for i, line in enumerate(test_lines):
            tokens = self.tokenize(line)
            ids = self.encode(line)
            total_tokens += len(ids)
            unk_count += sum(1 for tid in ids if tid == unk_id)

            reconstructed = self.decode(ids)
            if reconstructed == line:
                exact_matches += 1

            if i < 15:
                print(f"[Line {i+1:2d}] Text: {line}")
                print(f"          Tokens ({len(tokens)}): [{' | '.join(tokens)}]")
                print(f"          IDs: {ids}")
                print(f"          Reconstructed: {reconstructed} ({'MATCH OK' if reconstructed == line else 'FAIL'})\n")

        compression = total_chars / max(1, total_tokens)
        fertility = total_tokens / max(1, len(test_lines))
        acc = (exact_matches / max(1, len(test_lines))) * 100.0

        print("---------------- Evaluation Summary ----------------")
        print(f"Total Test Entries       : {len(test_lines)}")
        print(f"Total Original Chars     : {total_chars}")
        print(f"Total Tokens Produced    : {total_tokens}")
        print(f"Exact Reconstruction     : {exact_matches} / {len(test_lines)} ({acc:.2f}%)")
        print(f"Compression Ratio        : {compression:.3f} chars/token")
        print(f"Fertility Rate           : {fertility:.3f} tokens/word")
        print(f"Unknown Tokens (<unk>)   : {unk_count}")
        print("========================================================\n")


if __name__ == "__main__":
    train_file = "data/train_dataset.txt"
    test_file = "data/test_dataset.txt"
    output_dir = "model"

    tok = BalochiBPETokenizer()
    tok.train(train_file, vocab_size=2000, min_frequency=2)
    tok.save(output_dir)
    tok.evaluate(test_file)
