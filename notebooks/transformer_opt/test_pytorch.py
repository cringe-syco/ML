"""
Async encoding with NATIVE backend (no ONNX).
Compare performance vs ONNX backend.
"""
# import os
# print(os.listdir())

from sentence_transformers import SentenceTransformer
from pathlib import Path
from time import time
import numpy as np
import os
import asyncio
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import pandas as pd
from functools import partial

no_of_threads = 4
no_of_workers = 3
batch_size = 256

# 12-core optimization for native backend
os.environ['OMP_NUM_THREADS'] = f"{no_of_threads}"
os.environ['MKL_NUM_THREADS'] = f"{no_of_threads}"
os.environ['TORCH_NUM_THREADS'] = f"{no_of_threads}"

# PyTorch threading
torch.set_num_threads(no_of_threads)
torch.set_num_interop_threads(1)

print("System Configuration (Native Backend - 12-core):")
print(f"  PyTorch threads: {no_of_threads}")
print(f"  PyTorch interop threads: 1")
print(f"  OMP/MKL threads: {no_of_threads}")
print(f"  Backend: Native (PyTorch)")
print(f"  Async: ENABLED\n")


def get_sample_data(n=100000):
    """Generate sample sentences."""
    texts = [
        "The system processes user authentication requests securely.",
        "Machine learning models are trained on large datasets.",
        "Cloud computing enables scalable infrastructure.",
        "Data science explores patterns in datasets.",
        "Embeddings represent text as vectors.",
    ]
    df = pd.read_parquet(r"D:\Dsoft\Projects\ML\notebooks\ultrachat_200k_sft.parquet")  # Load data to ensure it's cached
    texts = df["prompt"].tolist()
    texts = texts[:n]
    del df
    # return (texts * (n // len(texts) + 1))[:n]
    return texts


# class NativeAsyncEncoder:
#     """Async encoder using native PyTorch backend."""
    
#     def __init__(self,   model_name=r'notebooks/model_files', num_workers= no_of_workers):
#         # Load with native backend (no ONNX)
#         self.model = SentenceTransformer(model_name)
#         # Ensure model is on CPU
#         # self.model = self.model.to('cpu')
#         self.executor = ThreadPoolExecutor(max_workers=num_workers)
#         self.loop = asyncio.get_event_loop()
    
#     async def encode_async(self, text_chunks, batch_size=batch_size):
#         """Encode text chunks asynchronously with native backend."""
#         tasks = [
#             self.loop.run_in_executor(
#                 self.executor,
#                 lambda chunk=chunk: self.model.encode(
#                     chunk,
#                     batch_size=batch_size,
#                     show_progress_bar=False,
#                     convert_to_numpy=True,
#                     device='cpu'
#                 )
#             )
#             for chunk in text_chunks
#         ]
        
#         results = await asyncio.gather(*tasks)
#         return np.vstack(results)

def encode_chunk(model, chunk, batch_size):
    """Helper function to encode a chunk of text."""
    return model.encode(
        chunk,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        device='cpu'
    )

class NativeAsyncEncoder:
    """Async encoder using native PyTorch backend."""
    
    def __init__(self, model_name=r'notebooks/model_files', num_workers=no_of_workers):
        # Load with native backend (no ONNX)
        self.model = SentenceTransformer(model_name)
        # Ensure model is on CPU
        # self.model = self.model.to('cpu')
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.loop = asyncio.get_event_loop()
    
    async def encode_async(self, text_chunks, batch_size=batch_size):
        """Encode text chunks asynchronously with native backend."""
        # Partial function to pass necessary parameters to encode_chunk
        tasks = [
            self.loop.run_in_executor(
                self.executor,
                partial(encode_chunk, self.model, chunk, batch_size)
            )
            for chunk in text_chunks
        ]
        
        results = await asyncio.gather(*tasks)
        return np.vstack(results)

async def main():
    print("=" * 70)
    print("NATIVE BACKEND + ASYNC ENCODING (12-CORE OPTIMIZED)")
    print("=" * 70)
    
    # Load data
    texts = get_sample_data(100000)
    print(f"\nProcessing {len(texts)} sentences...\n")
    
    # Initialize encoder with native backend
    print("Loading model with NATIVE backend (PyTorch)...")
    encoder = NativeAsyncEncoder('notebooks/model_files', num_workers=3)
    print("✓ Model loaded\n")
    
    # Split into chunks for parallel processing
    chunk_size = len(texts) // no_of_workers
    text_chunks = [
        texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)
    ]
    
    print("Async encoding with NATIVE backend (4 chunks, 4 workers)...")
    start = time()

    monitor_task = asyncio.create_task(monitor_cpu())
    embeddings = await encoder.encode_async(text_chunks, batch_size=batch_size)
    monitor_task.cancel()

    elapsed = time() - start
    throughput = len(texts) / elapsed
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS (NATIVE BACKEND + ASYNC)")
    print("=" * 70)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:,.0f} sentences/sec")
    print(f"Output shape: {embeddings.shape}")
    print(f"Precision: {embeddings.dtype}")
    
    print(f"\nOptimizations active:")
    print(f"  ✓ {no_of_workers} concurrent encoding tasks (ThreadPoolExecutor)")
    print(f"  ✓ Each task gets {no_of_threads // no_of_workers} cores ({no_of_threads}÷{no_of_workers} = {no_of_threads // no_of_workers} cores/task)")
    print(f"  ✓ PyTorch num_threads={no_of_threads}")
    print(f"  ✓ PyTorch interop_threads=1")
    print(f"  ✓ Batch size {batch_size} for efficient CPU utilization")
    print(f"  ✓ Device: CPU")
    print(f"  ✓ Backend: Native (not ONNX)")
    
    # Compare to ONNX
    onnx_throughput = 1303  # from previous run
    current_throughput = throughput
    comparison = current_throughput / onnx_throughput
    
    print(f"\n" + "=" * 70)
    print("NATIVE vs ONNX COMPARISON")
    print("=" * 70)
    print(f"ONNX throughput:    1,303 sent/s (baseline)")
    print(f"Native throughput:  {current_throughput:,.0f} sent/s")
    
    if comparison > 1:
        print(f"Native is {comparison:.2f}x FASTER than ONNX ✓")
    else:
        print(f"ONNX is {1/comparison:.2f}x faster than Native")
    
    
    return embeddings


async def monitor_cpu(interval=2):
    """Background task to print CPU stats while encoding."""
    while True:
        # Get per-core usage to see if all 12 cores are busy
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        avg_usage = sum(per_core) / len(per_core)
        print(f"  [CPU Monitor] Total: {avg_usage:.1f}% | Per-Core: {per_core}")
        await asyncio.sleep(interval)

if __name__ == "__main__":
    # Run async main
    embeddings = asyncio.run(main())
    # embeddings = await main()
    print(f"\n✓ Encoding complete!")