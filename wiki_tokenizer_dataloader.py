"""
Text File Tokenizer + PyTorch DataLoader for LLM Training
==========================================================
Pipeline:
  1. Read raw text from a local .txt file
  2. Clean Wikipedia boilerplate / nav noise
  3. Tokenize with tiktoken (BPE tokenizer, same family as GPT/Claude)
  4. Split tokens into train / dev / test
  5. Wrap each split in a TokenDataset
  6. Feed into DataLoaders that yield (input, target) batches

Install dependencies:
    pip install torch tiktoken
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# --------------------------------------------------
# 1.  READ + CLEAN TEXT
# --------------------------------------------------

def read_text_file(path: str) -> str:
    """Read and return the contents of a plain-text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """
    Remove common Wikipedia copy-paste noise:
      - Navigation / UI chrome  (Edit, Talk, Log in, Contents ...)
      - Reference list entries  (^ Author, Year. Title ...)
      - Bare citation markers   ([1], [2][3] on their own line)
      - Inline citation tags    ([1] embedded mid-sentence)
      - Excess blank lines

    No external libraries required -- stdlib re only.
    """

    SKIP_PATTERNS = [
        r"^(Article|Talk|Read|Edit|View history|Tools|"
        r"From Wikipedia.*|WikipediaThe Free Encyclopedia|"
        r"Wikipedia The Free Encyclopedia|"
        r"Donate|Create account|Log in|Contents|"
        r"move to sidebar|hide|Top|\(Top\)|"
        r"Jump to navigation|Jump to search|Navigation menu|"
        r"Main page|See also|References|External links|"
        r"Further reading|Categories:.*|Retrieved from.*|"
        r"This page was last edited.*|Text is available under.*|"
        r"Privacy policy|About Wikipedia|Disclaimers|"
        r"Contact Wikipedia|Cookie statement|Mobile view|"
        r"Wikimedia Foundation|Powered by MediaWiki|"
        r"Wikimedia Commons.*|Official website.*|vte|"
        r"Toggle.*subsection|Pages.*edited.*recently)$",
        r"^\^",
        r"^\s*(\[\d+\])+\s*$",
        r"^\s*[\d\W]+\s*$",
    ]

    skip_re = re.compile("|".join(SKIP_PATTERNS), re.IGNORECASE)
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if skip_re.match(stripped):
            continue
        stripped = re.sub(r"\[\d+\]", "", stripped)
        stripped = re.sub(r"\[citation needed\]", "", stripped, flags=re.IGNORECASE)
        stripped = stripped.strip()
        if stripped:
            cleaned.append(stripped)

    return "\n".join(cleaned).strip()


# --------------------------------------------------
# 2.  TOKENIZER  (tiktoken BPE)
# --------------------------------------------------

def get_tokenizer(encoding: str = "cl100k_base"):
    """
    cl100k_base  -> GPT-4 vocabulary (~100k tokens)
    p50k_base    -> GPT-3 vocabulary (~50k tokens)
    r50k_base    -> GPT-2/Codex vocabulary
    """
    return tiktoken.get_encoding(encoding)


def tokenize(text: str, enc) -> list[int]:
    """Convert raw text into a flat list of integer token IDs."""
    return enc.encode(text)


# --------------------------------------------------
# 3.  TRAIN / DEV / TEST SPLIT
# --------------------------------------------------

def split_tokens(
    token_ids: list[int],
    train_ratio: float = 0.8,
    dev_ratio:   float = 0.1,
    # test_ratio is whatever remains (default 0.1)
) -> tuple[list[int], list[int], list[int]]:
    """
    Split a flat token list into train / dev / test by ratio.

    Splits are contiguous (not shuffled) so the model sees coherent
    text within each split -- shuffling at the token level would break
    sentence context.  Shuffling happens later inside the DataLoader
    at the *window* level.

    Default split: 80% train | 10% dev | 10% test

    Args:
        token_ids   : full flat list of token IDs
        train_ratio : fraction for training   (0.0 - 1.0)
        dev_ratio   : fraction for validation (0.0 - 1.0)
                      test gets the remainder

    Returns:
        (train_ids, dev_ids, test_ids)
    """
    assert train_ratio + dev_ratio < 1.0, \
        "train_ratio + dev_ratio must be less than 1.0"

    n = len(token_ids)
    train_end = int(n * train_ratio)
    dev_end   = int(n * (train_ratio + dev_ratio))

    train_ids = token_ids[:train_end]
    dev_ids   = token_ids[train_end:dev_end]
    test_ids  = token_ids[dev_end:]

    return train_ids, dev_ids, test_ids


# --------------------------------------------------
# 4.  DATASET  (sliding-window chunks)
# --------------------------------------------------

class TokenDataset(Dataset):
    """
    Splits a flat token sequence into overlapping windows of length
    `context_length`.  Each item returns:
        x  -- input tokens  (indices 0 ... context_length-1)
        y  -- target tokens (indices 1 ... context_length)  <- shifted by 1

    Args:
        token_ids      : flat list of integer token IDs
        context_length : tokens per training example (e.g. 128, 512, 1024)
        stride         : tokens to advance between windows.
                         stride == context_length -> non-overlapping (default)
                         stride == 1             -> maximum overlap
    """

    def __init__(self, token_ids: list[int], context_length: int = 128,
                 stride: int | None = None):
        if stride is None:
            stride = context_length

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        self.indices = list(range(0, len(self.tokens) - context_length, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        end   = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


# --------------------------------------------------
# 5.  DATALOADER factory
# --------------------------------------------------

def build_dataloader(token_ids: list[int], context_length: int = 128,
                     stride: int | None = None, batch_size: int = 32,
                     shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Returns a DataLoader that yields (x, y) batches of shape
        [batch_size, context_length].

    shuffle=True  for train
    shuffle=False for dev / test
    """
    dataset = TokenDataset(token_ids, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# --------------------------------------------------
# 6.  DEMO
# --------------------------------------------------

def main():
    FILE_PATH      = "anthropic.txt"   # <- your file here
    CONTEXT_LENGTH = 128
    STRIDE         = 64
    BATCH_SIZE     = 16
    TRAIN_RATIO    = 0.99              # 99%  train
    DEV_RATIO      = 0.005             # 0.5% dev  -- test gets remaining 0.5%

    # Read
    print(f"Reading file: '{FILE_PATH}' ...")
    raw_text = read_text_file(FILE_PATH)
    print(f"  Characters (raw)     : {len(raw_text):,}")

    # Clean
    text = clean_text(raw_text)
    print(f"  Characters (cleaned) : {len(text):,}")
    print(f"  Preview              : {text[:120]!r}")

    # Tokenize
    enc = get_tokenizer("cl100k_base")
    token_ids = tokenize(text, enc)
    print(f"  Tokens (total)       : {len(token_ids):,}")

    # Split
    train_ids, dev_ids, test_ids = split_tokens(
        token_ids, train_ratio=TRAIN_RATIO, dev_ratio=DEV_RATIO
    )
    print(f"\nSplit ({int(TRAIN_RATIO*100)}/{int(DEV_RATIO*100)}/{int((1-TRAIN_RATIO-DEV_RATIO)*100)}):")
    print(f"  Train tokens         : {len(train_ids):,}")
    print(f"  Dev   tokens         : {len(dev_ids):,}")
    print(f"  Test  tokens         : {len(test_ids):,}")

    # Build DataLoaders
    #   train  -> shuffle=True  (randomise window order each epoch)
    #   dev    -> shuffle=False (deterministic evaluation)
    #   test   -> shuffle=False (deterministic evaluation)
    train_loader = build_dataloader(train_ids, CONTEXT_LENGTH, STRIDE,
                                    BATCH_SIZE, shuffle=True)
    dev_loader   = build_dataloader(dev_ids,   CONTEXT_LENGTH, STRIDE,
                                    BATCH_SIZE, shuffle=False)
    test_loader  = build_dataloader(test_ids,  CONTEXT_LENGTH, STRIDE,
                                    BATCH_SIZE, shuffle=False)

    print(f"\nDataLoader batches:")
    print(f"  Train                : {len(train_loader):,}")
    print(f"  Dev                  : {len(dev_loader):,}")
    print(f"  Test                 : {len(test_loader):,}")

    # Verify shapes from each loader
    for name, loader in [("train", train_loader),
                         ("dev",   dev_loader),
                         ("test",  test_loader)]:
        x_batch, y_batch = next(iter(loader))
        print(f"\n  [{name}] x: {x_batch.shape}  y: {y_batch.shape}  "
              f"dtype: {x_batch.dtype}")

    print("\nDone -- plug train_loader / dev_loader / test_loader into your loop.")
    print("""
Typical training loop skeleton:
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, y in dev_loader:   # <- monitor val loss here
                ...
""")


if __name__ == "__main__":
    main()