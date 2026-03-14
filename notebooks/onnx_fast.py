"""
Fast ONNX encoding optimization for 12-core systems.
Tests GPU acceleration, batch size tuning, and worker optimization.
"""

import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from pathlib import Path
from time import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import torch

# Environment setup
os.environ['ORT_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

print("=" * 70)
print("FAST ENCODING OPTIMIZATION FOR 12-CORE SYSTEM")
print("=" * 70)

# Check GPU availability
gpu_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
print(f"\nGPU Available: {gpu_available} ({gpu_name})")
print(f"CUDA Version: {torch.version.cuda if gpu_available else 'N/A'}\n")

# Session options
options = ort.SessionOptions()
options.intra_op_num_threads = 12
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

model_path = Path(__file__).parent / "model_files"

# ============================================================================
# STRATEGY 1: GPU ACCELERATION (if available)
# ============================================================================
if gpu_available:
    print("=" * 70)
    print("STRATEGY 1: GPU ACCELERATION")
    print("=" * 70)
    try:
        print("Loading with GPU backend (onnx_cuda)...")
        model_gpu = SentenceTransformer(str(model_path), device='cuda', backend='onnx_cuda')
        
        # Sample data
        from onnx_test import get_data
        data = get_data()
        test_data = (data * 50)[:50000]  # 50k for faster testing
        
        print(f"Testing {len(test_data)} sentences on GPU...\n")
        
        start = time()
        embeddings_gpu = model_gpu.encode(
            test_data,
            batch_size=512,
            show_progress_bar=False,
            convert_to_numpy=False  # Keep on GPU longer
        )
        elapsed_gpu = time() - start
        throughput_gpu = len(test_data) / elapsed_gpu
        
        print(f"GPU Results:")
        print(f"  Time: {elapsed_gpu:.2f}s")
        print(f"  Throughput: {throughput_gpu:,.0f} sentences/sec")
        print(f"  Speedup vs CPU (635): {throughput_gpu/635:.1f}x\n")
        
    except Exception as e:
        print(f"GPU acceleration failed: {e}\n")

# ============================================================================
# STRATEGY 2: OPTIMIZED CPU - LARGER BATCHES WITH FEWER WORKERS
# ============================================================================
print("=" * 70)
print("STRATEGY 2: OPTIMIZED CPU - LARGE BATCH + 2 WORKERS")
print("=" * 70)

model_cpu = SentenceTransformer(str(model_path), backend='onnx')

from onnx_test import get_data
data = get_data()
test_data = (data * 167)[:100000]  # Full 100k

print(f"Testing {len(test_data)} sentences on CPU...\n")

def encode_chunk_large(chunk, batch_size=2048):
    """Encode with larger batch size."""
    return model_cpu.encode(
        chunk,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )

# Test with 2 workers (6 cores each) and batch size 2048
print("Configuration: 2 workers × 6 cores, batch size 2048")
start = time()
chunks = np.array_split(test_data, 2)
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(lambda c: encode_chunk_large(c, batch_size=2048), chunks))
embeddings = np.vstack(results)
elapsed_2w = time() - start
throughput_2w = len(test_data) / elapsed_2w

print(f"\nCPU Results (2 workers):")
print(f"  Time: {elapsed_2w:.2f}s")
print(f"  Throughput: {throughput_2w:,.0f} sentences/sec")
print(f"  Speedup vs baseline (635): {throughput_2w/635:.1f}x\n")

# ============================================================================
# STRATEGY 3: SINGLE THREAD + ULTRA LARGE BATCH
# ============================================================================
print("=" * 70)
print("STRATEGY 3: SINGLE THREAD - ULTRA LARGE BATCH (4096)")
print("=" * 70)

print(f"Configuration: 1 worker, batch size 4096 (all 12 cores)")
start = time()
embeddings_single = model_cpu.encode(
    test_data,
    batch_size=4096,
    show_progress_bar=False,
    convert_to_numpy=True
)
elapsed_1w = time() - start
throughput_1w = len(test_data) / elapsed_1w

print(f"\nCPU Results (single thread):")
print(f"  Time: {elapsed_1w:.2f}s")
print(f"  Throughput: {throughput_1w:,.0f} sentences/sec")
print(f"  Speedup vs baseline (635): {throughput_1w/635:.1f}x\n")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

results_summary = [
    ("Baseline (4 workers, batch=1024)", 635),
    ("2 workers, batch=2048", throughput_2w),
    ("1 worker, batch=4096", throughput_1w),
]

if gpu_available:
    results_summary.insert(0, ("GPU (onnx_cuda, batch=512)", throughput_gpu))

results_summary.sort(key=lambda x: x[1], reverse=True)

print("\nPerformance Ranking:")
for i, (method, throughput) in enumerate(results_summary, 1):
    speedup = throughput / 635
    print(f"  {i}. {method:45s} {throughput:7,.0f} sent/s ({speedup:.1f}x)")

best_method, best_throughput = results_summary[0]
print(f"\n✓ RECOMMENDED: {best_method}")
print(f"  Expected speedup: {best_throughput/635:.1f}x faster than baseline")

# Code recommendation
if "GPU" in best_method:
    print(f"\nCode to use:")
    print(f"  model = SentenceTransformer(model_path, device='cuda')")
    print(f"  embeddings = model.encode(texts, batch_size=512)")
elif "1 worker" in best_method:
    print(f"\nCode to use:")
    print(f"  embeddings = model.encode(texts, batch_size=4096)")
else:
    print(f"\nCode to use:")
    print(f"  def encode_fast(texts):")
    print(f"      chunks = np.array_split(texts, 2)")
    print(f"      with ThreadPoolExecutor(max_workers=2) as executor:")
    print(f"          results = list(executor.map(")
    print(f"              lambda c: model.encode(c, batch_size=2048),")
    print(f"              chunks")
    print(f"          ))")
    print(f"      return np.vstack(results)")
    print(f"  embeddings = encode_fast(texts)")
