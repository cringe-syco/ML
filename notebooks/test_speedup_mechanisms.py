"""
Test precision-preserving speedup mechanisms.
No quantization, no precision loss - pure optimization.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from time import time
from sentence_transformers import SentenceTransformer
import torch
import os

# 12-core setup
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['TORCH_NUM_THREADS'] = '12'
torch.set_num_threads(12)
torch.set_num_interop_threads(1)


def get_sample_data(n=100000):
    """Generate varied-length sample sentences (realistic data)."""
    texts = [
        "Short text.",
        "This is a medium length sentence with more information.",
        "The quick brown fox jumps over the lazy dog and continues moving through the forest very quickly.",
        "A",  # Very short
        "Long text " * 50,  # Very long
    ]
    data = (texts * (n // len(texts) + 1))[:n]
    return data


def smart_chunk_by_chars(texts, num_workers=4):
    """Split into chunks balanced by CHARACTER count (not just count)."""
    total_chars = sum(len(t) for t in texts)
    target_chars = total_chars / num_workers
    
    chunks = []
    current_chunk = []
    current_chars = 0
    
    for text in texts:
        current_chunk.append(text)
        current_chars += len(text)
        
        if current_chars >= target_chars:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Pad to num_workers chunks
    while len(chunks) < num_workers:
        chunks.append([])
    
    return chunks[:num_workers]


def equal_chunk(texts, num_workers=4):
    """Split into equal-count chunks (current method)."""
    chunk_size = len(texts) // num_workers
    return [
        texts[i:i+chunk_size] 
        for i in range(0, len(texts), chunk_size)
    ][:num_workers]


class OptimizedEncoder:
    """Flexible encoder for testing configurations."""
    
    def __init__(self, model_path='notebooks/model_files', num_workers=4):
        self.model = SentenceTransformer(model_path)
        self.model = self.model.to('cpu')
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.loop = asyncio.get_event_loop()
        self.num_workers = num_workers
    
    async def encode_async(self, text_chunks, batch_size=1024, dynamic_batch=False):
        """Encode with optional dynamic batching."""
        tasks = []
        
        for chunk in text_chunks:
            if dynamic_batch:
                # Adjust batch size based on chunk size
                if len(chunk) < 1000:
                    bsize = 512
                elif len(chunk) > 50000:
                    bsize = 2048
                else:
                    bsize = 1024
            else:
                bsize = batch_size
            
            task = self.loop.run_in_executor(
                self.executor,
                lambda c=chunk, b=bsize: self.model.encode(
                    c,
                    batch_size=b,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device='cpu'
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return np.vstack([r for r in results if len(r) > 0])


async def benchmark_config(texts, config):
    """Benchmark a specific configuration."""
    name = config['name']
    workers = config['workers']
    batch_size = config['batch_size']
    chunking = config['chunking']
    dynamic_batch = config.get('dynamic_batch', False)
    
    print(f"\n{'=' * 70}")
    print(f"Config: {name}")
    print(f"  Workers: {workers}")
    print(f"  Batch size: {batch_size}" + (" (dynamic)" if dynamic_batch else ""))
    print(f"  Chunking: {chunking}")
    print(f"{'=' * 70}")
    
    # Chunk data
    if chunking == 'smart':
        chunks = smart_chunk_by_chars(texts, num_workers=workers)
        print(f"  Smart chunks: {[len(c) for c in chunks]}")
    else:
        chunks = equal_chunk(texts, num_workers=workers)
        print(f"  Equal chunks: {[len(c) for c in chunks]}")
    
    # Encode
    encoder = OptimizedEncoder(num_workers=workers)
    
    start = time()
    embeddings = await encoder.encode_async(
        chunks, 
        batch_size=batch_size,
        dynamic_batch=dynamic_batch
    )
    elapsed = time() - start
    
    throughput = len(texts) / elapsed
    
    # Results
    print(f"\n  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} sent/sec")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    
    return {
        'name': name,
        'throughput': throughput,
        'time': elapsed,
        'workers': workers,
        'batch_size': batch_size,
        'chunking': chunking,
        'embeddings': embeddings
    }


async def main():
    print("\n" + "=" * 70)
    print("PRECISION-PRESERVING SPEEDUP TESTING")
    print("=" * 70)
    print("Testing configurations to find optimal settings")
    print("No quantization • No precision loss • float32 preserved\n")
    
    # Load data
    texts = get_sample_data(100000)
    print(f"Sample texts: {len(texts)}")
    print(f"Length range: {min(len(t) for t in texts)} - {max(len(t) for t in texts)} chars")
    print(f"Total chars: {sum(len(t) for t in texts):,}")
    
    # Test configurations
    configs = [
        # Baseline: current setup
        {
            'name': 'BASELINE (current)',
            'workers': 4,
            'batch_size': 1024,
            'chunking': 'equal'
        },
        
        # Test 1: Smart chunking
        {
            'name': 'Smart chunking',
            'workers': 4,
            'batch_size': 1024,
            'chunking': 'smart'
        },
        
        # Test 2: Worker count variations
        {
            'name': 'Workers=2 (equal)',
            'workers': 2,
            'batch_size': 1024,
            'chunking': 'equal'
        },
        {
            'name': 'Workers=3 (equal)',
            'workers': 3,
            'batch_size': 1024,
            'chunking': 'equal'
        },
        {
            'name': 'Workers=6 (equal)',
            'workers': 6,
            'batch_size': 1024,
            'chunking': 'equal'
        },
        
        # Test 3: Dynamic batch sizing
        {
            'name': 'Dynamic batching',
            'workers': 4,
            'batch_size': 1024,
            'chunking': 'equal',
            'dynamic_batch': True
        },
        
        # Test 4: Combined approach
        {
            'name': 'OPTIMIZED (smart+dynamic)',
            'workers': 4,
            'batch_size': 1024,
            'chunking': 'smart',
            'dynamic_batch': True
        },
    ]
    
    # Run benchmarks
    results = []
    for config in configs:
        result = await benchmark_config(texts, config)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL CONFIGURATIONS")
    print("=" * 70)
    
    print(f"\n{'Configuration':<30} {'Throughput':<20} {'Time':<10} {'vs Baseline'}")
    print("-" * 70)
    
    baseline_throughput = results[0]['throughput']
    
    for result in results:
        speedup = (result['throughput'] / baseline_throughput - 1) * 100
        sign = "+" if speedup > 0 else ""
        color = "↑" if speedup > 0 else "↓" if speedup < 0 else "="
        
        print(
            f"{result['name']:<30} "
            f"{result['throughput']:>8,.0f} sent/s  "
            f"{result['time']:>8.2f}s  "
            f"{color} {sign}{speedup:>6.1f}%"
        )
    
    # Recommendation
    best = max(results, key=lambda x: x['throughput'])
    gain = ((best['throughput'] / baseline_throughput) - 1) * 100
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\n🏆 Best configuration: {best['name']}")
    print(f"   Throughput: {best['throughput']:,.0f} sent/sec")
    print(f"   Speedup over baseline: {gain:+.1f}%")
    print(f"\nApply by:")
    print(f"   workers={best['workers']}")
    print(f"   batch_size={best['batch_size']}")
    print(f"   chunking={best['chunking']}")
    if best.get('dynamic_batch'):
        print(f"   dynamic_batch=True")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted")
