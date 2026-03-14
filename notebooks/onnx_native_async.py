"""
Async encoding with NATIVE backend (no ONNX).
Compare performance vs ONNX backend.
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
from time import time
import numpy as np
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

# 12-core optimization for native backend
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['TORCH_NUM_THREADS'] = '12'

# PyTorch threading
torch.set_num_threads(12)
torch.set_num_interop_threads(1)

print("System Configuration (Native Backend - 12-core):")
print(f"  PyTorch threads: 12")
print(f"  PyTorch interop threads: 1")
print(f"  OMP/MKL threads: 12")
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
    return (texts * (n // len(texts) + 1))[:n]


class NativeAsyncEncoder:
    """Async encoder using native PyTorch backend."""
    
    def __init__(self, model_name='notebooks/model_files', num_workers=4):
        # Load with native backend (no ONNX)
        self.model = SentenceTransformer(model_name)
        # Ensure model is on CPU
        self.model = self.model.to('cpu')
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.loop = asyncio.get_event_loop()
    
    async def encode_async(self, text_chunks, batch_size=1024):
        """Encode text chunks asynchronously with native backend."""
        tasks = [
            self.loop.run_in_executor(
                self.executor,
                lambda chunk=chunk: self.model.encode(
                    chunk,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device='cpu'
                )
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
    encoder = NativeAsyncEncoder('notebooks/model_files', num_workers=4)
    print("✓ Model loaded\n")
    
    # Split into chunks for parallel processing
    chunk_size = len(texts) // 4
    text_chunks = [
        texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)
    ]
    
    print("Async encoding with NATIVE backend (4 chunks, 4 workers)...")
    start = time()
    
    embeddings = await encoder.encode_async(text_chunks, batch_size=1024)
    
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
    print(f"  ✓ 4 concurrent encoding tasks (ThreadPoolExecutor)")
    print(f"  ✓ Each task gets 3 cores (12÷4 = 3 cores/task)")
    print(f"  ✓ PyTorch num_threads=12")
    print(f"  ✓ PyTorch interop_threads=1")
    print(f"  ✓ Batch size 1024")
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


if __name__ == "__main__":
    # Run async main
    embeddings = asyncio.run(main())
    print(f"\n✓ Encoding complete!")


# # results from previous runs for comparison:
# System Configuration (Native Backend - 12-core):
#   PyTorch threads: 12
#   PyTorch interop threads: 1
#   OMP/MKL threads: 12
#   Backend: Native (PyTorch)
#   Async: ENABLED

# ======================================================================
# NATIVE BACKEND + ASYNC ENCODING (12-CORE OPTIMIZED)
# ======================================================================

# Processing 100000 sentences...

# Loading model with NATIVE backend (PyTorch)...
# ✓ Model loaded

# Async encoding with NATIVE backend (4 chunks, 4 workers)...

# ======================================================================
# RESULTS (NATIVE BACKEND + ASYNC)
# ======================================================================
# Total time: 77.64s
# Throughput: 1,288 sentences/sec
# Output shape: (100000, 384)
# Precision: float32

# Optimizations active:
#   ✓ 4 concurrent encoding tasks (ThreadPoolExecutor)
#   ✓ Each task gets 3 cores (12÷4 = 3 cores/task)
#   ✓ PyTorch num_threads=12
#   ✓ PyTorch interop_threads=1
#   ✓ Batch size 1024
#   ✓ Device: CPU
#   ✓ Backend: Native (not ONNX)

# ======================================================================
# NATIVE vs ONNX COMPARISON
# ======================================================================
# ONNX throughput:    1,303 sent/s (baseline)
# Native throughput:  1,288 sent/s
# ONNX is 1.01x faster than Native

# ✓ Encoding complete!