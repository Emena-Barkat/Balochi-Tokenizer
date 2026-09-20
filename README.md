# Balochi-Tokenizer

A subword tokenizer for the **Balochi** language, built with **Byte-Pair Encoding (BPE)** and trained from scratch on a Balochi word-list dataset.

Developed as an internship project at **Balochi Academy**. The datasets were provided by Balochi Academy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emena-Barkat/Balochi-Tokenizer/blob/main/balochi_tokenizer_colab.ipynb)

> **Status:** research project in progress. The accompanying research paper will be added to the `docs/` folder once it is finalized (see [Research paper](#research-paper)).

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Repository structure](#repository-structure)
3. [Requirements](#requirements)
4. [How to use it](#how-to-use-it)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Notes and limitations](#notes-and-limitations)
8. [Research paper](#research-paper)
9. [References](#references)
10. [Author, license and contact](#author-license-and-contact)

---

## What this project does

A *tokenizer* splits text into small pieces (called *tokens*) and turns those pieces into numbers, so that a computer program can work with the text. This project provides a BPE tokenizer for Balochi:

- **Training:** learns BPE merge rules from a training text file, starting from single characters. Default settings are a target vocabulary size of **2000** and a minimum pair frequency of **2**.
- **Special tokens:** `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`.
- **Encoding / decoding:** converts text to token strings and token IDs, and converts IDs back to text.
- **Saving / loading:** saves the trained tokenizer to a folder (`vocab.json`, `merges.txt`, `tokenizer_config.json`; the PowerShell version also writes `merges.json` and `tokenizer.json`).
- **Evaluation:** measures, on a test file, the exact-reconstruction rate, compression ratio (characters per token), fertility (tokens per line) and the number of unknown (`<unk>`) tokens.

The same tokenizer is provided in three forms:

| File | What it is |
| ---- | ---------- |
| `tokenizer.py` | Python version (`BalochiBPETokenizer` class). Uses only the Python standard library. |
| `balochi_tokenizer_colab.ipynb` | Google Colab notebook: upload the data, train, evaluate, try your own text. |
| `balochi_tokenizer.ps1` | Windows PowerShell version (the tokenizer engine is written in C# inside the script). |

**How the tokenizer treats text:** it does *not* split text on spaces first. Each line of the training file is treated as one unit and broken into characters, and merges are learned on those characters. The dataset used here is mostly one word per line (see [Dataset](#dataset)).

---

## Repository structure

```text
Balochi-Tokenizer/
├── README.md                        <- this file
├── .gitignore                       <- tells Git which files to leave out
├── tokenizer.py                     <- Python tokenizer (train / encode / decode / evaluate)
├── balochi_tokenizer_colab.ipynb    <- Google Colab notebook
├── balochi_tokenizer.ps1            <- Windows PowerShell version
├── data/
│   ├── train_dataset.txt            <- training data (one entry per line)
│   └── test_dataset.txt             <- test data (one entry per line)
├── model/                           <- trained tokenizer files
│   ├── vocab.json
│   ├── merges.txt
│   ├── merges.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── results/                         <- (to be added) evaluation outputs
└── docs/                            <- (to be added) research paper, after it is finalized
```

`results/` and `docs/` do not exist yet. They will be added when there are real results and a finished paper.

---

## Requirements

- **Python 3** for `tokenizer.py`. No extra packages are needed (it only uses the standard library).
- **Google account** if you want to use the Colab notebook.
- **Windows PowerShell** if you want to use `balochi_tokenizer.ps1`.

---

## How to use it

Run all commands from the main project folder (the folder that contains `tokenizer.py`).

### Option 1 — Python script

Train the tokenizer, save it to `model/`, and evaluate it on the test file:

```bash
python tokenizer.py
```

The script reads `data/train_dataset.txt` and `data/test_dataset.txt`, trains with `vocab_size=2000` and `min_frequency=2`, saves the files to `model/`, and prints an evaluation summary.

Use the tokenizer from your own Python code:

```python
from tokenizer import BalochiBPETokenizer

tok = BalochiBPETokenizer()
tok.train("data/train_dataset.txt", vocab_size=2000, min_frequency=2)

text = "جلشکاناں"
tokens = tok.tokenize(text)   # list of subword pieces
ids = tok.encode(text)        # list of token IDs
print(tokens)                 # ['جل', 'شک', 'اناں']
print(ids)                    # [1508, 300, 791]
print(tok.decode(ids))        # جلشکاناں
```

(The tokens and IDs shown are from one training run with the settings above.)

### Option 2 — Google Colab

1. Click the **Open In Colab** badge at the top of this page (or upload `balochi_tokenizer_colab.ipynb` to Colab yourself).
2. Run the cells from top to bottom.
3. When asked, upload `train_dataset.txt` and `test_dataset.txt` from the `data/` folder.

The notebook trains the tokenizer, evaluates it on the test file, lets you try your own Balochi text, and can zip the trained `model` folder for download.

### Option 3 — Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\balochi_tokenizer.ps1 -TrainPath data\train_dataset.txt -TestPath data\test_dataset.txt
```

Available options (from the script's parameters):

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `-Mode` | `TrainAndTest` | `TrainAndTest`, `TrainOnly`, or `TestOnly` (test an already-trained model in `model/`) |
| `-TrainPath` | `train_dataset.txt` | Training file |
| `-TestPath` | `test_dataset.txt` | Test file |
| `-OutputDir` | `model` | Folder where the model is saved / loaded from |
| `-VocabSize` | `2000` | Target vocabulary size |
| `-MinFrequency` | `2` | Minimum pair frequency for a merge |
| `-CustomText` | (empty) | Tokenize one piece of text and print tokens, IDs and the reconstructed text |

---

## Dataset

| File | Non-empty lines | Unique lines |
| ---- | --------------- | ------------ |
| `data/train_dataset.txt` | 77,804 | 33,994 |
| `data/test_dataset.txt` | 8,645 | 5,773 |

- Each line is one entry. Almost all entries are single words; a small number of lines contain more than one word.
- The training text contains **150** distinct characters, which form the base alphabet of the tokenizer.
- A small number of lines contain English (Latin-letter) words: 543 lines in the training file and 75 in the test file. They were not removed.
- Source of the data: provided by Balochi Academy.
- Dataset license / permission to share: to be confirmed with Balochi Academy.

---

## Results

Output of `python tokenizer.py` on the included datasets with the default settings (`vocab_size=2000`, `min_frequency=2`):

| Item | Value |
| ---- | ----- |
| Vocabulary size | 2000 (5 special tokens + 150 characters + 1845 merges) |
| Test entries | 8,645 |
| Total characters in test data | 60,224 |
| Total tokens produced | 21,292 |
| Exact reconstruction | 8,645 / 8,645 (100.00%) |
| Compression ratio | 2.828 characters per token |
| Fertility | 2.463 tokens per entry (entries are almost all single words) |
| Unknown tokens (`<unk>`) | 0 |

To reproduce these numbers, run `python tokenizer.py` and read the "Evaluation Summary" printed at the end.

**How to read these numbers:** the tokenizer can always fall back to single characters, and every character in the training data is in the vocabulary. So exact reconstruction is expected to be 100% unless the test text contains characters that never appeared in training. It shows that encoding followed by decoding gives back the original text. It does **not** measure how linguistically good the subword splits are.

No comparison with other tokenizers and no downstream-task results are reported here yet.

---

## Notes and limitations

- **Train/test overlap:** 3,013 of the 5,773 unique test entries (about 52%) also appear in the training file. Keep this in mind when interpreting the test results.
- **Training data is word-list style:** the tokenizer was trained on mostly single-word lines. Behaviour on full sentences has not been evaluated here.
- **Known issues in `tokenizer.py` loading (`load()`):**
  - It cannot read files that start with a UTF-8 byte-order mark (BOM). The files currently in `model/` start with one, so `load("model")` fails with a `JSONDecodeError`.
  - Merge rules that contain the space character are skipped when loading, so 5 of the 1,845 rules are dropped. This only affects text that contains spaces.
  - Training and using the tokenizer in one run (as in the example above) is not affected.

---

## Research paper

The research paper for this project is **not available yet**. It will be added to the `docs/` folder (for example `docs/research_paper.pdf`) once it is finalized.

- Title, status and link: to be added.

---

## References

- - Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. *Neural Machine Translation of Rare Words with Subword Units.* In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016), pages 1715–1725. This paper adapted the byte pair encoding compression algorithm for splitting words into subword units. https://aclanthology.org/P16-1162/

**How to cite this project:** to be added after the paper is available.

---

## Author, license and contact

- Author: Emena Barkat
- Developed as an internship project at Balochi Academy.
- GitHub: [Emena-Barkat](https://github.com/Emena-Barkat)
- Code license: not specified yet.
