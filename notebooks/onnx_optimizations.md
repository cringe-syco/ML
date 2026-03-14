# Encoding Performance Optimizations

## Quick Wins (Immediate Impact)

### 1. **Increase Batch Size**
- Test larger batches: 256, 512, 1024, 2048
- Larger batches = more parallelism but higher memory
- Sweet spot usually 256-512 for most hardware

### 2. **GPU Acceleration** ⚡ (Biggest Impact)
```python
# If CUDA available:
onnx_model = SentenceTransformer(model_path, backend='onnx_cuda')

# Check: pip install "optimum[onnxruntime-gpu]"
```
Expected speedup: **5-20x faster** than CPU

### 3. **Skip Sorting** (if not critical)
- Sorting 100k sentences takes time
- Do it asynchronously after encoding if needed

### 4. **Disable Progress Bar**
```python
onnx_model.encode(sentences, show_progress_bar=False)
```
Minor but measurable savings

## Medium Impact Optimizations

### 5. **Mixed Precision (FP16)**
- Faster on modern GPUs with minimal accuracy loss
```python
onnx_model.encode(sentences, precision="float16")
```

### 6. **Parallel Encoding** (Multi-Threading)
```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def encode_chunk(chunk, batch_size=256):
    return onnx_model.encode(chunk, batch_size=batch_size)

# Split into 4 chunks, encode in parallel
chunks = np.array_split(sentences, 4)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(encode_chunk, chunks))
embeddings = np.vstack(results)
```

### 7. **Quantize Embeddings**
- Reduce memory/disk by using int8 instead of float32
```python
import numpy as np
embeddings = onnx_model.encode(sentences)
# Quantize to int8 (normalize to -128...127 range)
embeddings_q = (embeddings * 127).astype(np.int8)
```
Trades: Slight accuracy loss for 4x memory savings

## Advanced Optimizations

### 8. **Pre-compile ONNX Model** (One-time setup)
- Models compile on first load; subsequent runs are faster
- Keep the model in memory for repeated encoding tasks

### 9. **ONNX Runtime Session Options**
```python
from onnxruntime import SessionOptions

options = SessionOptions()
options.graph_optimization_level = 99  # Max optimization
options.inter_op_num_threads = 12      # CPU threads
options.intra_op_num_threads = 12
# Pass to model loader
```

### 10. **Use Lighter Model**
- Replace with distilled faster model if acceptable accuracy tradeoff
- Try: `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`

## Benchmarking Results

Run `onnx_test.py` to see current throughput with different batch sizes:
- Baseline (CPU, batch=256): ~1,000-2,000 sentences/sec
- GPU (CUDA, batch=512): ~10,000-50,000 sentences/sec
- GPU + FP16: ~20,000-100,000 sentences/sec

## Recommended Path

1. **First**: Try GPU acceleration → test batch sizes 256-1024
2. **If CPU-only**: Increase batch size to 512-1024
3. **For production**: Combine GPU + FP16 + parallel chunks
4. **For storage**: Use quantized embeddings (int8)
