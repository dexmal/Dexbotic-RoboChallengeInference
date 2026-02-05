import numpy as np
from loguru import logger
import torch
from transformers import AutoTokenizer
from dexbotic.tokenization.process import DM0Tokenization
from .base_policy import BasePolicy


class DM0ProgPolicy(BasePolicy):
    """DM0 Prog model policy with progress embedding support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_progress = True
        
    def _load_model(self, ckpt_path: str) -> None:
        from dexbotic.model.dm0.dm0_prog_arch import DM0ProgForCausalLM
        logger.info("Loading DM0 Prog Architecture")
        
        try:
            model = DM0ProgForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load model with device_map: {e}")
            model = DM0ProgForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)

        # Setup Tokenizer (load from ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 200  # Prog needs longer seq len
        
        self.model = model
        self.tokenization_func = DM0Tokenization(self.tokenizer)
        logger.info("DM0 Prog Model loaded successfully")
        
        # Setup Transforms
        self._setup_transforms()

    def reset(self):
        self.progress = np.zeros((1, 1, 1), dtype=np.float32)
        logger.info(f"Reset progress, current progress is {self.progress}")

    def _hook_prepare_extra_inputs(self, inference_args, raw_state):
        inference_args["progress"] = self.progress
        return inference_args

    def _infer_implementation(self, inputs):
        actions, progress = self.model.inference_action(**inputs)
        return actions, progress

    def _hook_process_extra_outputs(self, inputs, actions, progress):
        if progress is not None:
            progress = progress.detach().cpu().numpy()
            # Update progress logic (from ProgPolicy)
            new_progress = self.progress + (progress - self.progress) * 1.0 / actions.shape[1] * self.action_horizon
            new_progress = np.clip(new_progress, 0.0, 1.0)
            logger.info(f"Progress: {self.progress} -> {new_progress}")
            self.progress = new_progress
