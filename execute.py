import os
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from policies import get_policy
from runner import InferenceRunner
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from utils.constants import TASK_METADATA, IMAGE_TYPE_MAP


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Get prompt and robot_type from TASK_METADATA
    if cfg.task_name not in TASK_METADATA:
        raise ValueError(f"Unknown task: {cfg.task_name}. Available: {list(TASK_METADATA.keys())}")
    
    metadata = TASK_METADATA[cfg.task_name]
    prompt = metadata["prompt"]
    robot_type = metadata["robot_type"]
    
    # Determine image_type and action_type
    image_type = IMAGE_TYPE_MAP[robot_type]
    action_type = "leftpos" if robot_type != "aloha" else "joint"


    # Setup log directory (from config or default)
    log_base_dir = cfg.get("log_dir", "./logs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_base_dir, cfg.task_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "runtime.log"), level="DEBUG", enqueue=False)
    
    logger.info(f"Task: {cfg.task_name}")
    logger.info(f"Checkpoint: {cfg.checkpoint}")
    logger.info(f"Robot type: {robot_type}")
    logger.info(f"Log dir: {log_dir}")
    
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
    
    # Online mode
    logger.info("Starting Online mode...")
    user_id = cfg.get("user_id", "")
    job_collection_id = cfg.get("job_collection_id", "")
    if not user_id:
        raise ValueError("user_id is required for online mode")
    client = InterfaceClient(user_id)
    job_loop(
        client, runner, job_collection_id,
        cfg.image_size, image_type, action_type, cfg.duration
    )


if __name__ == "__main__":
    main()
