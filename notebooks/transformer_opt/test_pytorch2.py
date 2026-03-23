from sentence_transformers import SentenceTransformer
from time import time
import numpy as np
import os
import torch
import pandas as pd

# 12-core optimization for native backend
no_of_threads = 12  # Use all available cores for the single process
batch_size = 256

os.environ['OMP_NUM_THREADS'] = f"{no_of_threads}"
os.environ['MKL_NUM_THREADS'] = f"{no_of_threads}"
os.environ['TORCH_NUM_THREADS'] = f"{no_of_threads}"

torch.set_num_threads(no_of_threads)
torch.set_num_interop_threads(1)

def get_sample_data(n=100000):
    """Generate sample sentences."""
    # Note: Ensure the path is correct for your local machine
    df = pd.read_parquet(r"D:\Dsoft\Projects\ML\notebooks\ultrachat_200k_sft.parquet")
    texts = df["prompt"].tolist()[:n]
    return texts

def main():
    print("=" * 70)
    print("NATIVE BACKEND - SEQUENTIAL PROCESSING")
    print("=" * 70)
    
    # 1. Load data
    texts = get_sample_data(100000)
    print(f"\nLoaded {len(texts)} sentences.")
    
    # 2. Initialize encoder
    print("Loading model...")
    model = SentenceTransformer('notebooks/model_files', device='cpu')
    print("✓ Model loaded\n")
    
    # 3. Sequential Encoding
    print(f"Encoding {len(texts)} sentences sequentially (Batch Size: {batch_size})...")
    start = time()
    
    # In sequential mode, we pass the whole list to the model
    # SentenceTransformer handles internal batching automatically
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time() - start
    throughput = len(texts) / elapsed
    
    # 4. Results
    print("\n" + "=" * 70)
    print("RESULTS (SEQUENTIAL)")
    print("=" * 70)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:,.0f} sentences/sec")
    print(f"Output shape: {embeddings.shape}")
    
    return embeddings

if __name__ == "__main__":
    embeddings = main()
    print(f"\n✓ Encoding complete!")
