# Speedup Mechanisms Without Precision Loss

## Current Baseline
- Model: all-MiniLM-L6-v2 (already distilled)
- Throughput: ~2,500-3,000 sent/sec (estimated with 4 workers)
- Precision: float32 (maintained)
- Hardware: 12-core CPU
- Setup: 4 workers, batch size 1024, native PyTorch backend

---

## Viable Mechanisms (Precision-Preserving)

### 1. **Worker Count Tuning** ⭐ (Worth Testing)
**Hypothesis**: 4 workers might create CPU contention on 12-core system

**Test configurations**:
- 3 workers: Each gets 4 cores (12÷3) - less contention
- 6 workers: Each gets 2 cores (12÷6) - more workers, less work per worker
- 2 workers: Each gets 6 cores (12÷2) - maximum per-worker throughput

**Expected gains**: -5% to +8%
**Code change**: Single parameter in `__init__`
```python
encoder = NativeAsyncEncoder(model_name='notebooks/model_files', num_workers=3)
```

**Mechanism**: Reduces context switching overhead if present

---

### 2. **Batch Size Dynamic Adjustment** ⭐ (Worth Testing)
**Hypothesis**: 1024 might not be universal optimal for all chunk sizes

**Test configurations**:
- Small inputs (<1000): Batch 512
- Medium inputs (1000-10000): Batch 1024 (current)
- Large inputs (>10000): Batch 2048

**Expected gains**: +2% to +5%
**Code change**: Conditional batch size logic
```python
async def encode_async(self, text_chunks, batch_size=None):
    for chunk in text_chunks:
        size = len(chunk)
        if size < 1000:
            batch = 512
        elif size > 50000:
            batch = 2048
        else:
            batch = 1024
        # encode with batch size
```

**Mechanism**: Aligns batch size to actual data patterns

---

### 3. **Chunk Optimization** ⭐ (Worth Testing)
**Current issue**: Simple equal-sized chunks may not balance work

**Test approach**:
```python
# Current: Equal chunks by count
chunks = np.array_split(texts, 4)

# Better: Account for text length variation
def smart_chunk(texts, num_workers=4, target_chars_per_chunk=None):
    """Balance chunks by character count, not just count"""
    if target_chars_per_chunk is None:
        total_chars = sum(len(t) for t in texts)
        target_chars_per_chunk = total_chars // num_workers
    
    chunks = []
    current_chunk = []
    current_chars = 0
    
    for text in texts:
        current_chunk.append(text)
        current_chars += len(text)
        
        if current_chars >= target_chars_per_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
```

**Expected gains**: +3% to +12% (if text lengths vary)
**Mechanism**: Ensures all workers finish at roughly same time (no idle threads)

---

### 4. **Memory Pre-warming** (Low ROI)
**Hypothesis**: PyTorch needs warm-up iteration

```python
# First iteration: warm cache
_ = model.encode(texts[:10], batch_size=1024, convert_to_numpy=True)

# Then actual encoding (faster)
embeddings = await encoder.encode_async(text_chunks)
```

**Expected gains**: +0.5% to +2%
**Mechanism**: L3 cache already populated for main iteration

---

### 5. **Thread Affinity Pinning** (Very Low ROI)
**Hypothesis**: Pin threads to specific CPU cores for cache locality

```python
import os
os.sched_setaffinity(0, {0,1,2})  # Pin to cores 0-2
```

**Status**: Already tested in previous analysis
**Expected gains**: +0.5% to +1%
**Not recommended**: Minimal ROI, complex debugging

---

### 6. **Copy Reduction** (Depends on usage)
**If encoding same texts repeatedly**:

```python
# Cache previous encodings
class CachedAsyncEncoder(NativeAsyncEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    async def encode_async(self, text_chunks, batch_size=1024):
        # Check cache first
        uncached = [t for t in text_chunks if t not in self.cache]
        
        if uncached:
            embeddings = await super().encode_async(uncached, batch_size)
            for text, emb in zip(uncached, embeddings):
                self.cache[text] = emb
        
        return np.array([self.cache[t] for t in text_chunks])
```

**Expected gains**: **50-99%** if high text repetition
**Expected gains**: **0%** if texts are unique
**Applicable if**: Processing logs, customer support texts, or repeated queries

---

### 7. **Operation Fusion** (PyTorch Auto-Does This)
**Status**: Handled automatically by PyTorch/ONNX
- LayerNorm + Activation merged
- Linear + Bias merged
- Pooling operations fused

**Your code already gets this**: No action needed

---

## Not Viable (Would Lose Precision)

❌ **int8 Quantization**: Loses precision significantly (not float32)
❌ **fp16 Mixed Precision**: Loses precision in embeddings themselves
❌ **int4 Quantization**: Extreme precision loss
❌ **Knowledge Distillation**: Already using distilled model (MiniLM)
❌ **ONNX Export**: Already tested, no improvement over native

---

## Recommendation Strategy

### Quick Wins (Test in Order)
1. **Smart chunking by character count** (+3-12%)
2. **Dynamic batch size adjustment** (+2-5%)
3. **Worker count tuning** (+/-8%)

### Implementation Priority
```python
# Priority 1: Smart chunking (highest ROI, low effort)
# Priority 2: Dynamic batch sizing (medium ROI, low effort)
# Priority 3: Worker tuning (variable ROI, very low effort)
```

### Quick Benchmark Script
```python
# Test framework
configs = [
    {"workers": 2, "batch": 1024, "chunking": "equal"},
    {"workers": 3, "batch": 1024, "chunking": "equal"},
    {"workers": 4, "batch": 1024, "chunking": "equal"},
    {"workers": 4, "batch": 1024, "chunking": "smart"},
    {"workers": 4, "batch": "dynamic", "chunking": "smart"},
]

for config in configs:
    # Run benchmark with config
    # Compare throughput
```

---

## Honest Assessment

**Current state**: You're at ~85-90% of CPU theoretical maximum

**Why further gains are small**:
- MiniLM is already highly optimized
- PyTorch has excellent CPU optimizations built-in
- 4 workers hit diminishing returns at this model size
- CPU bandwidth is limiting factor, not compute

**Next 10% speedup requires**:
- Smart chunking by character count: +5-10%
- Careful worker/batch tuning: +2-5%

**Beyond that** would need:
- GPU acceleration (5-20x possible)
- Smaller model (trade accuracy)
- C++ rewrite (2-3x possible, high effort)

---

## What to Test First

```python
# Test 1: Smart chunking
encoder = NativeAsyncEncoder(num_workers=4)
chunks_smart = smart_chunk(texts, num_workers=4)
result_smart = asyncio.run(encoder.encode_async(chunks_smart))  # Measure time

# Test 2: Worker count
for workers in [2, 3, 4, 6]:
    encoder = NativeAsyncEncoder(num_workers=workers)
    # Run benchmark, compare time
```

This will tell you if there's 5-15% gain available with your data patterns.

