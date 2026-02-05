import os
import json
import numpy as np
import torch
import megfile
from PIL import Image
from loguru import logger
from abc import ABC, abstractmethod

from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import ActionDenorm, AbsoluteAction


class BasePolicy(ABC):
    """
    Abstract Base Policy class for DB0 inference.
    """

    def __init__(
        self,
        ckpt_path: str,
        prompt: str,
        robot_type: str,
        action_type: str,
        image_shape=(728, 728),
        action_horizon: int = 15,
        num_images: int = 3,
        non_delta_mask: list = None,
        task_name: str = "",
        **kwargs
    ):
        logger.info(f"Initializing {self.__class__.__name__}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.prompt = prompt
        self.robot_type = robot_type.lower()
        self.action_type = action_type
        self.image_shape = image_shape
        self.action_horizon = action_horizon
        self.num_images = num_images
        # For aloha (dual-arm), gripper positions are at index 6 and 13
        if non_delta_mask:
            self.non_delta_mask = non_delta_mask
        elif self.robot_type == "aloha":
            self.non_delta_mask = [6, 13]
        else:
            self.non_delta_mask = [6]
        self.task_name = task_name

        logger.info(f"Checkpoint path: {ckpt_path}")
        logger.info(f"Prompt: {self.prompt}")
        logger.info(f"Robot type: {self.robot_type}")
        logger.info(f"Action type: {self.action_type}")

        # These flags might be overridden by subclasses
        self.use_progress = False
        self.use_history = False

        self.reset()
        
        # Load normalization stats
        self.norm_stats = self._read_normalization_stats(ckpt_path)
        
        # Load model (implemented by subclasses or calling a shared helper)
        self._load_model(ckpt_path)

        # Warmup
        self.warmup()

    @abstractmethod
    def _load_model(self, ckpt_path: str) -> None:
        """Load the specific model architecture."""
        pass

    def _read_normalization_stats(self, ckpt_path: str):
        """Read normalization stats from checkpoint directory."""
        norm_stats_file = os.path.join(ckpt_path, "norm_stats.json")
        logger.info(f"Reading normalization stats from {norm_stats_file}")
        if norm_stats_file is None or not megfile.smart_exists(norm_stats_file):
            logger.warning(f"Norm stats file not found at {norm_stats_file}, using default [-1, 1]")
            return {"min": -1, "max": 1}
        with megfile.smart_open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def _setup_transforms(self):
        """Build input/output transforms. Should be called after model load."""
        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False, use_quantiles=True),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False, use_quantiles=True),
                AbsoluteAction(),
            ]
        )

    def warmup(self):
        """Warmup the model with dummy inputs."""
        logger.info("Warming up...")
        dummy_images = {
            "image_0": np.zeros((self.image_shape[0], self.image_shape[1], 3)).astype(np.uint8),
            "image_1": np.zeros((self.image_shape[0], self.image_shape[1], 3)).astype(np.uint8),
            "image_2": np.zeros((self.image_shape[0], self.image_shape[1], 3)).astype(np.uint8),
        }
        state_dim = 14 if self.robot_type == "aloha" else 7
        dummy_state = np.zeros(state_dim)
        self._infer(dummy_images, dummy_state)
        logger.info("Warmup complete")


    def _prepare_inference_inputs(self, images: dict, state: np.ndarray):
        # Convert numpy images to PIL Images
        pil_images = []
        # Support variable number of images, but ordered by key usually
        for key in ["image_0", "image_1", "image_2"]:
            if key in images:
                pil_images.append(Image.fromarray(images[key]).convert("RGB"))
        
        # Process images
        batch_images_tensor = self.model.process_images(pil_images).to(dtype=self.model.dtype)
        
        # Pad/Truncate image batch
        num_inputs = len(pil_images)
        if num_inputs != self.num_images:
            if num_inputs < self.num_images:
                 # Pad with zeros
                padding = torch.zeros_like(batch_images_tensor[0:1]).repeat(
                    self.num_images - num_inputs, 1, 1, 1
                )
                batch_images_tensor = torch.cat([batch_images_tensor, padding], dim=0)
            else:
                batch_images_tensor = batch_images_tensor[:self.num_images]

        batch_image_masks = torch.tensor(
            [True] * num_inputs + [False] * (self.num_images - num_inputs),
            device=batch_images_tensor.device,
        )
        
        # Tokenize (Common)
        prompt_conv = [
            {"from": "human", "value": self.prompt},
            {"from": "gpt", "value": ""},
        ]
        batch_input_ids = np.array([self.tokenization_func(prompt_conv)["input_ids"]])
        batch_attention_mask = np.array(
            [np.array(ids != self.tokenizer.pad_token_id) for ids in batch_input_ids]
        )

        # Prepare state
        batch_states = state[None] if state.ndim == 1 else state
        batch_states = np.array(batch_states, dtype=np.float32)

        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor.unsqueeze(0), # Add batch dim
            "image_masks": batch_image_masks.unsqueeze(0),
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(self.non_delta_mask),
            },
        }
        
        # Hook for subclasses to add extra inputs (progress, history)
        inference_args = self._hook_prepare_extra_inputs(inference_args, state)
        
        # Apply transforms
        inputs = self.input_transform(inference_args)
        
        # Hook for subclasses to modify inputs after transform (e.g. History)
        inputs = self._hook_post_transform_inputs(inputs)
        
        # Ensure 'states' key exists for model compatibility (db0_arch expects 'states')
        if "states" not in inputs and "state" in inputs:
            inputs["states"] = inputs["state"]

        # Handle Discrete State Input special logic
        # Removed: Logic moved to StatePolicy


        # Final tensor conversion
        inputs["input_ids"] = torch.tensor(inputs["input_ids"], device=self.device)
        inputs["attention_mask"] = torch.tensor(batch_attention_mask, device=self.device)
        
        final_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        return final_inputs

    def _hook_prepare_extra_inputs(self, inference_args, raw_state):
        """Hook for subclasses to modify inference arguments before transform."""
        return inference_args

    def _hook_post_transform_inputs(self, inputs):
        """Hook for subclasses to modify inputs after transform."""
        return inputs

    def _infer(self, images: dict, state: np.ndarray) -> np.ndarray:
        # Capture original state dimension (e.g. 7 for Franka, 14 for Aloha)
        original_state_dim = state.shape[-1]
        # Store as instance variable for hooks to access before transform values
        self._original_state_dim = original_state_dim
        
        inputs = self._prepare_inference_inputs(images, state)
        
        # Model specific inference call
        actions, extra_outputs = self._infer_implementation(inputs)
        
        # Process output
        return self._process_model_output(inputs, actions, extra_outputs, original_state_dim)


    def infer(self, images: dict, state: np.ndarray, non_delta_mask: list = None) -> np.ndarray:
        """
        Public inference interface for InferenceRunner
        
        Args:
            images: image dictionary {image_key: numpy array}
            state: preprocessed state vector
            non_delta_mask: list of gripper position indices
        
        Returns:
            Raw action output (before postprocessing)
        """
        if non_delta_mask:
            self.non_delta_mask = non_delta_mask
        return self._infer(images, state)

    @abstractmethod
    def _infer_implementation(self, inputs):
        """
        Execute model inference. Implemented by subclass.
        Returns (actions, extra_outputs).
        """
        pass


    def _process_model_output(self, inputs, actions, extra_outputs, original_state_dim):
        """
        Process model output using original_state_dim to slice action.
        """
        # Hook for history/progress updates
        self._hook_process_extra_outputs(inputs, actions, extra_outputs)
        
        # Prepare for Denorm
        outputs = {
            k: v.to(torch.float32) if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v
            for k, v in inputs.items()
        }
        outputs = {
             k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
             for k, v in outputs.items()
        }
        outputs["action"] = actions.detach().cpu().numpy()
        
        # Denorm
        outputs = self.output_transform(outputs)
        
        # Slice back to original dimension (e.g. 7) BEFORE tricks
        output_action = outputs["action"][0, ..., :original_state_dim]
        logger.debug(f"output_action:\n{output_action}")
        
        return output_action

    def _hook_process_extra_outputs(self, inputs, actions, extra_outputs):
        pass

    def reset(self):
        """Reset internal state (history, progress)."""
        pass

    def reproduce(self, inputs):
        """Reproduce inference from saved inputs."""
        device_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        actions, extra = self._infer_implementation(device_inputs)
        # Try to infer original dim if possible, otherwise rely on inputs state dim (which is padded 32)
        # However, for reproduction/debug, typically strict correctness of dim might be less critical 
        # or we accept 32 dim. But let's try to be smart.
        # If 'raw_state' is unavailable here, we default to padded dim.
        state_dim = device_inputs["state"].shape[-1]
        return self._process_model_output(device_inputs, actions, extra, state_dim)
