# mmdet3d/utils/hooks/cuda_cache_clear_hook.py
from mmcv.runner import HOOKS, Hook
import torch, gc


@HOOKS.register_module()
class CudaCacheClearHook(Hook):
    def _clear(self, tag):
        print(f"[CudaCacheClearHook] {tag} - Clearing CUDA cache")
        # Show memory before
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3

        # Clear
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete

        # Show memory after
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3

        print(f"  Memory: {allocated_before:.2f}GB -> {allocated_after:.2f}GB (allocated)")
        print(f"  Memory: {reserved_before:.2f}GB -> {reserved_after:.2f}GB (reserved)")

    def after_val_epoch(self, runner):
        # Critical: clear after validation to prevent OOM in next epoch
        self._clear("after_val_epoch")

    def after_train_epoch(self, runner):
        # Clear after checkpoint saving
        self._clear("after_train_epoch")

    def before_train_epoch(self, runner):
        # Clear before starting next epoch (most important for preventing OOM)
        self._clear("before_train_epoch")

    def before_val_epoch(self, runner):
        # Also clear before validation
        self._clear("before_val_epoch")