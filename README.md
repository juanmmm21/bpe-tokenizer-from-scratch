# BPE Tokenizer from Scratch

This subproject implements a byte-level Byte Pair Encoding (BPE) tokenizer built entirely from scratch in Python, with no external dependencies. It's the foundational component in the natural language processing (NLP) cycle, responsible for translating plain text into a sequence of numeric identifiers (tokens) that autoregressive models (like Transformers) can process and project into their embedding space.

## Technical Architecture and BPE Fundamentals

Unlike approaches based on simple Unicode characters or full-word tokens, this tokenizer implements a byte-level approach (Byte-Level BPE) inspired by the one used in modern architectures like GPT-2 and Llama.

### 1. Base Vocabulary Initialization (256)
The vocabulary is immutably initialized with the 256 possible bytes (0 to 255). This guarantees complete representation of any plain text stream, including special characters, accents, emojis, or non-standardized Unicode sequences, without ever generating out-of-vocabulary tokens (<unk>).
To avoid processing issues with strings containing terminal control characters and whitespace, bytes are reversibly mapped to a range of safe, printable Unicode characters.

### 2. Regex-Based Pre-tokenization
Before searching for byte merges, a regular expression filter (with Unicode support) is applied that segments the text into individual words, numbers, and punctuation marks:

```python
# Strict segmentation pattern
PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```

This prevents the BPE algorithm from merging characters across important semantic boundaries (for example, joining the last letter of a word with a space or a punctuation mark), which would produce a chaotic, redundant vocabulary.

### 3. Merge Learning Algorithm (Training)
During training, the algorithm operates iteratively over the input corpus represented as byte sequences:

1. The frequencies of all consecutive token pairs within the segmented words are computed.
2. The most frequent pair $(t_i, t_{j})$ is selected:
   
   $$(t_i, t_j) = \arg\max_{(x,y)} \text{freq}(x, y)$$
   
3. A new merge rule is added: $(t_i, t_j) \rightarrow t_{new}$.
4. The sequence is updated by replacing all consecutive occurrences of $t_i, t_j$ with $t_{new}$.
5. The process repeats until the vocabulary size reaches the preset limit `vocab_size` or no pairs remain with a frequency greater than 1.

### 4. Encoding Algorithm
To encode a test text string:
1. It's broken down into words using the pre-tokenization pattern.
2. Each word is converted to its byte representation mapped to safe characters.
3. The learned merge rules are applied greedily, strictly respecting the chronological order in which they were created during training.
4. The corresponding numeric IDs are returned.

### 5. Safe Decoding
Decoding retrieves the bytes corresponding to each vocabulary ID and concatenates them into a binary array. It's then decoded to UTF-8 using the error-replacement directive (`errors="replace"`) to mitigate exceptions on truncated sequences.

## Model Specification (JSON Schema)

The trained model is persisted as a JSON file with the following structure:

```json
{
  "vocab": {
    "h": 104,
    "o": 111,
    "l": 108,
    "a": 97,
    "ho": 256,
    "ol": 257,
    "hola": 258
  },
  "merges": [
    ["h", "o"],
    ["o", "l"],
    ["ho", "la"]
  ]
}
```

## Using it as a package

Since `0.1.0` the code lives in the `bpe_tokenizer_from_scratch` package and the repository is
pip-installable, so other projects can depend on it instead of copying files
or editing `sys.path`:

```bash
pip install "bpe-tokenizer-from-scratch @ git+https://github.com/juanmmm21/bpe-tokenizer-from-scratch.git@v0.1.0"
```

```python
from bpe_tokenizer_from_scratch import BPETokenizer
```

The old top-level modules are kept as thin re-exports, so scripts that add
this folder to `sys.path` keep working. This component imports no other
ai-core component: they are combined in the `aicore` SDK of
[ai-core-infra-hub](https://github.com/juanmmm21/ai-core-infra-hub), which
passes collaborators (vector stores, encoders, tokenizers) in by injection.

Development: `pip install -e ".[dev]"`, then `pytest` and `ruff check .`.

## Installation Requirements

*   Python 3.10 or higher.
*   No external dependencies.

## Execution and Verification Guide

### 1. Set Up Environment and Clone
```bash
git clone https://github.com/juanmmm21/bpe-tokenizer-from-scratch.git
cd bpe-tokenizer-from-scratch
```

### 2. Run Unit Tests
To verify mathematical consistency (e.g. the identity property: `decode(encode(text)) == text`), run:
```bash
python3 -m unittest test_tokenizer.py
```

### 3. Run the Demo
Start the interactive demo script to see initialization, training, emoji encoding, and vocabulary JSON import/export:
```bash
python3 example.py
```

## Connectivity within the ai-core-infra Ecosystem

The `bpe-tokenizer-from-scratch` tokenizer is the foundation of data processing in the infrastructure:
*   Its token IDs feed directly into [contrastive-embedding-trainer](https://github.com/juanmmm21/contrastive-embedding-trainer) for siamese embedding training.
*   It serves as the physical max_tokens limit in [semantic-chunking-engine](https://github.com/juanmmm21/semantic-chunking-engine).
*   The saved vocabulary and merges are loaded by [llm-inference-server](https://github.com/juanmmm21/llm-inference-server) for streaming token decoding.
