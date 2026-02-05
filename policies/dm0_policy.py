from loguru import logger
import torch
from transformers import AutoTokenizer
from dexbotic.tokenization.process import DM0Tokenization
from .base_policy import BasePolicy


class DM0Policy(BasePolicy):
    """DM0 model policy"""
    
    def _load_model(self, ckpt_path: str) -> None:
        from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
        logger.info("Loading DM0 Architecture")
        
        try:
            model = DM0ForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load model with device_map: {e}")
            model = DM0ForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)

        # Setup Tokenizer (load from ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 100
        
        self.model = model
        self.tokenization_func = DM0Tokenization(self.tokenizer)
        logger.info("DM0 Model loaded successfully")
        
        # Setup Transforms
        self._setup_transforms()

    def _infer_implementation(self, inputs):
        actions = self.model.inference_action(**inputs)
        return actions, None
