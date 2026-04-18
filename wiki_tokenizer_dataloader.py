"""
Text File Tokenizer + PyTorch DataLoader for LLM Training
==========================================================
Pipeline:
  1. Read raw text from a local .txt file
  2. Tokenize with tiktoken (BPE tokenizer, same family as GPT/Claude)
  3. Wrap in a custom torch.utils.data.Dataset
  4. Feed into a DataLoader that yields (input, target) batches

Install dependencies:
    pip install torch tiktoken
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# ──────────────────────────────────────────────
# 1.  READ TEXT  (local file)
# ──────────────────────────────────────────────

def read_text_file(path: str) -> str:
    """Read and return the contents of a plain-text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────
# 2.  TOKENIZER  (tiktoken BPE)
# ──────────────────────────────────────────────

def get_tokenizer(encoding: str = "cl100k_base"):
    """
    cl100k_base  → GPT-4 / text-embedding-ada-002 vocabulary (~100k tokens)
    p50k_base    → GPT-3 vocabulary (~50k tokens)
    r50k_base    → older GPT-2/Codex vocabulary
    """
    return tiktoken.get_encoding(encoding)


def tokenize(text: str, enc) -> list[int]:
    """Convert raw text into a flat list of integer token IDs."""
    return enc.encode(text)


# ──────────────────────────────────────────────
# 3.  DATASET  (sliding-window chunks)
# ──────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Splits a flat token sequence into overlapping windows of length
    `context_length`.  Each item returns:
        x  – input tokens  (indices 0 … context_length-1)
        y  – target tokens (indices 1 … context_length)   ← shifted by 1

    This is the standard causal-LM "predict next token" setup.

    Args:
        token_ids      : flat list / tensor of integer token IDs
        context_length : number of tokens per training example (e.g. 128, 512, 1024)
        stride         : how many tokens to advance between windows.
                         stride == context_length → non-overlapping (faster, less data)
                         stride == 1             → maximum overlap  (slower, more data)
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int = 128,
        stride: int | None = None,
    ):
        if stride is None:
            stride = context_length          # non-overlapping by default

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride

        # Pre-compute starting indices for every window
        self.indices = list(
            range(0, len(self.tokens) - context_length, stride)
        )

    # Required by PyTorch Dataset ──────────────

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end   = start + self.context_length

        x = self.tokens[start : end]       # input  sequence
        y = self.tokens[start + 1 : end + 1]  # target sequence (shifted right by 1)

        return x, y


# ──────────────────────────────────────────────
# 4.  DATALOADER  factory
# ──────────────────────────────────────────────

def build_dataloader(
    token_ids: list[int],
    context_length: int = 128,
    stride: int | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Returns a DataLoader that yields (x, y) batches of shape
        [batch_size, context_length]
    ready to plug straight into model training.

    Args:
        shuffle      : True for training, False for validation/test
        num_workers  : parallel data loading processes (set > 0 on Linux/Mac)
    """
    dataset = TokenDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # faster GPU transfer
        drop_last=True,    # discard the final incomplete batch
    )


# ──────────────────────────────────────────────
# 5.  DEMO  – run end-to-end
# ──────────────────────────────────────────────

def main():
    FILE_PATH      = "anthropic.txt"    # ← put your .txt file path here
    CONTEXT_LENGTH = 128              # tokens per training example
    STRIDE         = 64               # 50 % overlap between windows
    BATCH_SIZE     = 16

    # ── read ──
    print(f"Reading file: '{FILE_PATH}' …")
    text = read_text_file(FILE_PATH)
    print(f"  Characters read    : {len(text):,}")

    # ── tokenize ──
    enc = get_tokenizer("cl100k_base")
    token_ids = tokenize(text, enc)
    print(f"  Tokens             : {len(token_ids):,}")
    print(f"  First 10 token IDs : {token_ids[:10]}")
    print(f"  Decoded back       : {enc.decode(token_ids[:10])!r}")

    # ── dataset ──
    dataset = TokenDataset(token_ids, CONTEXT_LENGTH, STRIDE)
    print(f"\nNumber of data example      : {len(dataset):,}")

    x_sample, y_sample = dataset[0]
    print(f"  x[0] shape         : {x_sample.shape}   (input)")
    print(f"  y[0] shape         : {y_sample.shape}   (target, shifted +1)")
    print(f"  First input tokens : {x_sample[:8].tolist()}")
    print(f"  Decoded            : {enc.decode(x_sample[:8].tolist())!r}")

    # ── dataloader ──
    loader = build_dataloader(
        token_ids,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    print(f"\nDataLoader batches   : {len(loader):,}")

    # iterate one epoch
    for step, (x_batch, y_batch) in enumerate(loader):
        # x_batch shape: [batch_size, context_length]
        # y_batch shape: [batch_size, context_length]
        if step == 0:
            print(f"\n  Step 0 – x_batch : {x_batch.shape}")
            print(f"  Step 0 – y_batch : {y_batch.shape}")
            print(f"  dtype            : {x_batch.dtype}")

        # ── your model forward + loss goes here ──
        # logits = model(x_batch)
        # loss   = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
        # loss.backward()

    print(f"\n  Iterated {step + 1} batches  ✓")
    print("\nDone – plug `loader` into your training loop!")


if __name__ == "__main__":
    main()
