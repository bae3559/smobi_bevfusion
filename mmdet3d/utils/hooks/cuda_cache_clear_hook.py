# mmdet3d/utils/hooks/cuda_cache_clear_hook.py
from mmcv.runner import HOOKS, Hook
import torch, gc


@HOOKS.register_module()
class CudaCacheClearHook(Hook):
    def _clear(self, tag):
        # print(f"[CudaCacheClearHook] {tag}")  # 원하면 로깅
        gc.collect()
        torch.cuda.empty_cache()

    def after_val_epoch(self, runner):
        self._clear("after_val_epoch")

    def after_train_epoch(self, runner):
        # 체크포인트 저장 직후에 오는 훅. (EvalHook → CheckpointHook → after_train_epoch 순서가 일반적)
        self._clear("after_train_epoch")

    def before_train_epoch(self, runner):
        # 검증/저장 직후 다음 에폭 들어가기 직전에 한 번 더
        self._clear("before_train_epoch")