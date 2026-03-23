import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

########################################
# CONFIG
########################################

MODEL_PATH = r'D:\Dsoft\Projects\ML\notebooks\model_files'
DATA_PATH = r"D:\Dsoft\Projects\ML\notebooks\ultrachat_200k_sft.parquet"

BATCH_SIZE = 384
NUM_WORKERS = 12   # Use 12 out of 16 cores
MAX_SEQ_LEN = 128

########################################
# DATASET (with index tracking)
########################################

class IndexedDataset(Dataset):
    def __init__(self, texts):
        self.data = list(enumerate(texts))  # (idx, text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

########################################
# COLLATE (worker-safe tokenizer)
########################################

tokenizer = None

def collate_fn(batch):
    global tokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True
        )

    indices, texts = zip(*batch)

    features = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )

    return indices, features

########################################
# DATA LOADING
########################################

def get_sample_data(n=100_000):
    print(f"Reading {n} rows from parquet...")

    df = pd.read_parquet(
        DATA_PATH,
        engine='fastparquet',
        columns=["prompt"]
    )

    texts = df["prompt"].values[:n].tolist()

    # ⚡ Efficient bucketing without breaking order
    indexed = list(enumerate(texts))

    # sort by token length but keep index
    indexed.sort(key=lambda x: min(len(x[1].split()), MAX_SEQ_LEN))

    return indexed, len(texts)

########################################
# MAIN
########################################

def main():

    # CPU tuning
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Load data
    indexed_texts, total_size = get_sample_data(100_000)
    print(f"✓ Loaded {total_size} sentences.")

    dataset = IndexedDataset([t[1] for t in indexed_texts])

    # Load model
    print("Loading model to CUDA...")
    model = SentenceTransformer(MODEL_PATH, device='cuda')
    model.eval()
    print("✓ Model ready\n")

    loader = DataLoader(
        list(enumerate([t[1] for t in indexed_texts])),  # keep index
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Preallocate result buffer (order preserved)
    final_embeddings = [None] * total_size

    print(f"Running on {torch.cuda.get_device_name(0)}")

    with torch.inference_mode():
        for indices, features in tqdm(loader, desc="Encoding"):

            # Move to GPU (async)
            for k in features:
                features[k] = features[k].to('cuda', non_blocking=True)

            # Forward
            out = model(features)
            embeddings = out['sentence_embedding'].cpu()

            # Restore order
            for idx, emb in zip(indices, embeddings):
                final_embeddings[idx] = emb

    # Stack in correct order
    final_embeddings = torch.stack(final_embeddings)

    print(f"\n✅ Done! Shape: {final_embeddings.shape}")

    return final_embeddings


if __name__ == "__main__":
    main()