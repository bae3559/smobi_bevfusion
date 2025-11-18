"""Custom EvalHook that clears CUDA cache after evaluation"""

import torch
import gc
from mmdet.core import DistEvalHook, EvalHook


class EvalHookWithCacheClear(EvalHook):
    """EvalHook with CUDA cache clearing after evaluation"""

    def _do_evaluate(self, runner):
        """Perform evaluation and clear cache"""
        results = super()._do_evaluate(runner)

        # Clear CUDA cache after evaluation
        print("[EvalHookWithCacheClear] Clearing CUDA cache after evaluation")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return results


class DistEvalHookWithCacheClear(DistEvalHook):
    """DistEvalHook with CUDA cache clearing after evaluation"""

    def _do_evaluate(self, runner):
        """Perform evaluation and clear cache"""
        results = super()._do_evaluate(runner)

        # Clear CUDA cache after evaluation
        print("[DistEvalHookWithCacheClear] Clearing CUDA cache after evaluation")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return results
