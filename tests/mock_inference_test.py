"""
Mock Inference Test

Standalone test script extracted from execute.py mock mode.
Generates synthetic images and state data, then runs the full inference pipeline
to verify model loading, preprocessing, inference, and postprocessing.

Usage:
    python tests/mock_inference_test.py --config-name=specialist/put_cup_on_coaster
    python tests/mock_inference_test.py task_name=put_cup_on_coaster checkpoint=./checkpoints/DM0-table30_put_cup_on_coaster
"""

import io
import os
import sys
import datetime

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from PIL import Image

# Add project root to path so imports work when running from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policies import get_policy
from runner import InferenceRunner
from utils.constants import TASK_METADATA, IMAGE_TYPE_MAP


def generate_mock_state(robot_type: str, image_type: list, image_size: tuple) -> dict:
    """
    Generate mock state data for testing.

    Args:
        robot_type: Type of robot (aloha, arx5, ur5, franka)
        image_type: List of image source names (e.g. ["high", "left_hand", "right_hand"])
        image_size: Image dimensions (height, width)

    Returns:
        Dictionary with "images" (source -> PNG bytes) and "action" (state vector)
    """
    # Generate mock images
    mock_images = {}
    for img_source in image_type:
        img_array = np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        mock_images[img_source] = buffer.getvalue()

    # Generate mock state vector
    if robot_type == "aloha":
        # Joint: 7 joints * 2 arms = 14
        mock_action = np.random.randn(14).astype(np.float32)
    else:
        # EEF: pos(3) + quat(4) + gripper(1) = 8
        mock_action = np.random.randn(8).astype(np.float32)

    return {"images": mock_images, "action": mock_action}


def run_mock_test(cfg: DictConfig) -> None:
    """
    Run mock inference test with the given Hydra config.

    Args:
        cfg: Hydra DictConfig with task_name, checkpoint, etc.
    """
    # Resolve task metadata
    if cfg.task_name not in TASK_METADATA:
        raise ValueError(f"Unknown task: {cfg.task_name}. Available: {list(TASK_METADATA.keys())}")

    metadata = TASK_METADATA[cfg.task_name]
    prompt = metadata["prompt"]
    robot_type = metadata["robot_type"]
    image_type = IMAGE_TYPE_MAP[robot_type]
    action_type = "leftpos" if robot_type != "aloha" else "joint"

    logger.info(f"Task: {cfg.task_name}")
    logger.info(f"Checkpoint: {cfg.checkpoint}")
    logger.info(f"Robot type: {robot_type}")
    logger.info(f"Image type: {image_type}")
    logger.info(f"Action type: {action_type}")

    # Initialize Policy
    policy = get_policy(
        ckpt_path=cfg.checkpoint,
        policy_type=cfg.get("policy_type", None),
        prompt=prompt,
        robot_type=robot_type,
        action_type=action_type,
        action_horizon=cfg.action_horizon,
        task_name=cfg.task_name,
        image_shape=tuple(cfg.image_size),
    )

    # Initialize InferenceRunner
    runner = InferenceRunner(
        policy=policy,
        robot_type=robot_type,
        action_type=action_type,
        task_name=cfg.task_name,
        image_type=image_type,
        action_horizon=cfg.action_horizon,
        postprocess_args=OmegaConf.to_container(cfg.postprocess_args) if cfg.get("postprocess_args") else None,
    )

    # Generate mock data
    mock_state = generate_mock_state(robot_type, image_type, tuple(cfg.image_size))
    logger.info(f"Generated mock state: {len(mock_state['images'])} images, "
                f"action shape: {mock_state['action'].shape}")

    # Run inference
    logger.info("Running mock inference...")
    actions = runner.infer(mock_state)

    # Validate output
    logger.info(f"Mock inference completed! Output: {len(actions)} actions")
    if actions:
        logger.info(f"First action: {actions[0]}")
        logger.info(f"Action dimension: {len(actions[0])}")
    else:
        logger.warning("No actions returned!")

    logger.info("Mock inference test PASSED")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Setup log directory
    log_base_dir = cfg.get("log_dir", "./logs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_base_dir, "mock_test", cfg.task_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "mock_test.log"), level="DEBUG", enqueue=False)

    run_mock_test(cfg)


if __name__ == "__main__":
    main()
