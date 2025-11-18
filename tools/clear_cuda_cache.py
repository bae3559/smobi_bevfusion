#!/usr/bin/env python3
"""Clear CUDA cache and show memory usage"""

import torch
import gc

print("Before clearing:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Clear Python garbage
gc.collect()

# Clear CUDA cache
torch.cuda.empty_cache()

# Reset peak memory stats
torch.cuda.reset_peak_memory_stats()

print("\nAfter clearing:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\nCUDA cache cleared!")
