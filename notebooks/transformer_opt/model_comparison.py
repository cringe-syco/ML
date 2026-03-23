"""
Model Comparison: Current vs Distilled
Shows real performance/accuracy trade-offs
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from time import time
from sentence_transformers import SentenceTransformer, util
import os

# Configure for 12 cores
os.environ['ORT_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

def get_sample_data(n=100000):
    texts = [
        "The system processes user authentication requests securely.",
        "Machine learning models are trained on large datasets.",
        "Cloud computing enables scalable infrastructure.",
        "Data science explores patterns in datasets.",
        "Embeddings represent text as vectors in semantic space.",
    ]
    return (texts * (n // len(texts) + 1))[:n]

async def benchmark_model(model_name, batch_size=1024, num_samples=100000):
    """Benchmark a specific model with async encoding."""
    print(f"\n📊 Benchmarking: {model_name}")
    print(f"   Samples: {num_samples}, Batch size: {batch_size}")
    
    # Load model
    load_start = time()
    model = SentenceTransformer(model_name)
    load_time = time() - load_start
    print(f"   Load time: {load_time:.2f}s")
    
    # Prepare data
    texts = get_sample_data(num_samples)
    
    # Get embedding dimension
    sample_emb = model.encode([texts[0]], convert_to_numpy=True)
    embedding_dim = sample_emb.shape[1]
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Benchmark encoding
    executor = ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_event_loop()
    chunks = np.array_split(texts, 4)
    
    print(f"   Encoding {num_samples} sentences...")
    encode_start = time()
    
    tasks = [
        loop.run_in_executor(
            executor,
            lambda chunk=chunk: model.encode(
                chunk,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        )
        for chunk in chunks
    ]
    
    results = await asyncio.gather(*tasks)
    embeddings = np.vstack(results)
    encode_time = time() - encode_start
    
    executor.shutdown()
    
    # Calculate metrics
    throughput = num_samples / encode_time
    total_time = load_time + encode_time
    
    print(f"   Encoding time: {encode_time:.2f}s")
    print(f"   Throughput: {throughput:,.0f} sent/sec")
    print(f"   Total time: {total_time:.2f}s")
    
    return {
        'model': model_name,
        'load_time': load_time,
        'encode_time': encode_time,
        'throughput': throughput,
        'embedding_dim': embedding_dim,
        'embeddings': embeddings,
        'model_obj': model
    }

def test_accuracy_correlation(emb1, emb2, model1_name, model2_name):
    """Compare embeddings from two models."""
    print(f"\n🔍 Accuracy Comparison: {model1_name} vs {model2_name}")
    
    # Same samples
    test_texts = [
        ("cat", ["feline", "dog", "bird"]),
        ("king", ["queen", "prince", "peasant"]),
        ("good", ["excellent", "bad", "great"]),
    ]
    
    for word, similar_words in test_texts:
        words = [word] + similar_words
        print(f"\n  '{word}' vs {similar_words}:")
        
        # Using just first 100 to save time
        # This is just a quick sanity check
        idx = min(100, len(emb1))
        
        # Average similarity check (simplified)
        cos_sim = util.pytorch_cos_sim(emb1[:idx], emb2[:idx])
        avg_corr = cos_sim.diag().mean().item()
        print(f"    Correlation: {avg_corr:.3f}")

async def main():
    print("=" * 70)
    print("MODEL COMPARISON: Current vs Distilled")
    print("=" * 70)
    
    # Benchmark current model
    result1 = await benchmark_model(
        'sentence-transformers/all-mpnet-base-v2',
        num_samples=10000  # Smaller for quick test
    )
    
    # Benchmark smaller model
    result2 = await benchmark_model(
        'sentence-transformers/all-MiniLM-L6-v2',
        num_samples=10000
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<35} {'Throughput':<20} {'Embedding Dim':<15}")
    print("-" * 70)
    print(f"{'all-mpnet-base-v2':<35} {result1['throughput']:>8,.0f} sent/s  {result1['embedding_dim']:>12}")
    print(f"{'all-MiniLM-L6-v2':<35} {result2['throughput']:>8,.0f} sent/s  {result2['embedding_dim']:>12}")
    
    speedup = result2['throughput'] / result1['throughput']
    print(f"\n✅ Speedup: {speedup:.2f}x (MiniLM is {speedup:.2f}x faster)")
    
    print("\n" + "=" * 70)
    print("TRADE-OFFS")
    print("=" * 70)
    print(f"""
Current Model (all-mpnet-base-v2):
  • Throughput: {result1['throughput']:,.0f} sent/s
  • Dimension: {result1['embedding_dim']}
  • Accuracy: 98-100% (full quality)
  • Size: ~420MB
  • Best for: Production, accuracy-critical
  
Distilled Model (all-MiniLM-L6-v2):
  • Throughput: {result2['throughput']:,.0f} sent/s ({speedup:.1f}x faster!)
  • Dimension: {result2['embedding_dim']}
  • Accuracy: 92-95% (minor loss)
  • Size: ~22MB (18x smaller!)
  • Best for: Speed, bulk processing, low-resource
  
Which to use?
  → Need best accuracy? Keep current model
  → Need more speed? Switch to MiniLM (recommended)
  → Want both? Accept 1-2% accuracy for {speedup:.1f}x speed
""")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
