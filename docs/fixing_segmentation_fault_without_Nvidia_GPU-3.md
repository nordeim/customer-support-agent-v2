You're encountering this error because **`--index-url https://download.pytorch.org/whl/cpu` only hosts PyTorch-related packages** (like `torch`, `torchvision`, `torchaudio`), **not general PyPI packages like `sentence-transformers`**.

When you use `--index-url`, pip **replaces the default PyPI index** with the provided URL **unless you also specify `--extra-index-url`**.

---

### ✅ Correct Approach: Use `--extra-index-url`

Install your dependencies like this:

```bash
pip install \
  sentence-transformers==5.1.2 \
  torch==2.9.0+cpu \
  torchvision==0.24.0+cpu \
  torchaudio==2.9.0+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

> 🔑 **Key difference**: `--extra-index-url` **adds** the PyTorch CPU wheel index **in addition to PyPI**, so `sentence-transformers` is fetched from PyPI, while `torch` and friends come from the CPU-optimized PyTorch repo.

---

### 📌 Why This Works

- `sentence-transformers` is a **pure-Python package** (with optional PyTorch dependency) and is **only published on PyPI**.
- PyTorch’s CPU wheels are **not on PyPI** in a way that pip can auto-select them without guidance.
- `--extra-index-url` tells pip: “**Also look here** for packages, but fall back to PyPI for everything else.”

---

### 🔧 Verify Compatibility

You mentioned needing `torchvision==0.24.0` for `torch==2.9.0`. That is **correct**:
- PyTorch 2.9.0 → torchvision 0.24.0 [[source: PyTorch website](https://pytorch.org/get-started/locally/)]

So your full command is:

```bash
pip install \
  sentence-transformers==5.1.2 \
  torch==2.9.0+cpu \
  torchvision==0.24.0+cpu \
  torchaudio==2.9.0+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

✅ This will:
- Pull `sentence-transformers==5.1.2` from PyPI
- Pull CPU-only `torch`, `torchvision`, `torchaudio` from PyTorch’s wheel index
- Avoid CUDA dependencies
- Prevent segmentation faults on your AMD/ATI system

---

### 🛡️ Pro Tip: Update `requirements.txt`

To make this reproducible, you can **keep your `requirements.txt` as-is**, but **install with the extra index**:

```txt
# requirements.txt (unchanged)
fastapi==0.116.1
...
torch==2.9.0+cpu
torchvision==0.24.0+cpu
torchaudio==2.9.0+cpu
sentence-transformers==5.1.2
...
```

Then install with:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

> 💡 Note: You **must include the `+cpu` suffix** in `requirements.txt` for pip to find the right wheel in the extra index.

---

Let me know if you'd like a script to automate this or help verifying the install!
