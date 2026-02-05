import time
import cv2
import numpy as np
from loguru import logger
from typing import Dict, Any, List

from utils.constants import IMAGE_MAPPING, ALOHA_JOINT_MIN, ALOHA_JOINT_MAX
from utils.transforms import quat_to_euler, euler_to_quat, unwrap_euler_sequence


class InferenceRunner:
    """Inference runner: handles preprocessing, inference coordination, and postprocessing"""
    
    def __init__(
        self,
        policy,
        robot_type: str,
        action_type: str,
        task_name: str,
        image_type: List[str],
        action_horizon: int = 15,
        postprocess_args: dict = None,
    ):
        self.policy = policy
        self.robot_type = robot_type.lower()
        self.action_type = action_type
        self.task_name = task_name
        self.image_type = image_type
        self.action_horizon = action_horizon
        self.postprocess_args = postprocess_args or {}
        
        # robot_type -> image_key mapping
        self.image_mapping = IMAGE_MAPPING[self.robot_type]
        
        # gripper position indices
        self.non_delta_mask = [6, 13] if self.robot_type == "aloha" else [6]
        
        logger.info(f"InferenceRunner initialized: robot={robot_type}, task={task_name}")

    def reset_policy(self):
        self.policy.reset()

    def infer(self, state: Dict[str, Any]) -> List[float]:
        """
        Full inference pipeline: preprocess -> infer -> postprocess
        
        Args:
            state: dictionary containing "images" and "action"
            
        Returns:
            List of action sequences
        """
        # 1. Preprocess images
        images = self._parse_images(state["images"])
        
        # 2. Preprocess state
        processed_state = self._preprocess_state(state["action"])
        
        # 3. Model inference
        inference_start = time.time()
        raw_actions = self.policy.infer(images, processed_state, self.non_delta_mask)
        raw_actions = raw_actions[:self.action_horizon]
        logger.info(f"Inference time: {time.time() - inference_start:.4f}s")
        
        # 4. Postprocess
        actions = self._postprocess(raw_actions, processed_state)
        logger.debug(f"Actions after postprocess:\n{actions}")
        
        return actions.tolist()

    def _parse_images(self, images_dict: Dict[str, bytes]) -> Dict[str, np.ndarray]:
        """Parse images: PNG bytes -> numpy array"""
        result = {}
        for source in self.image_type:
            if source not in images_dict:
                continue
            image_data = images_dict[source]
            image = cv2.imdecode(
                np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_key = self.image_mapping.get(source)
            if image_key:
                result[image_key] = image
        return result

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Preprocess state: quat -> euler (if EEF control)"""
        if self.robot_type in ["franka", "ur5"] and self._is_eef_control():
            pos = state[:3]
            quat = state[3:7]
            gripper = state[7]
            euler = quat_to_euler(quat, degrees=False)
            return np.array([*pos, *euler, gripper], dtype=np.float32)
        return np.array(state, dtype=np.float32)

    def _is_eef_control(self) -> bool:
        """Check if using end-effector control"""
        return "pos" in self.action_type

    def _postprocess(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Postprocess: apply tricks"""
        actions = self._gripper_trick(actions)

        if self._is_eef_control():
            actions = self._eef_trick(actions, state)
            actions = self._action_euler_to_quat(actions)
        else:
            actions = self._joint_trick(actions)
        return actions

    def _action_euler_to_quat(self, actions: np.ndarray) -> np.ndarray:
        """Convert Euler angles in actions to quaternions"""
        if self.robot_type in ["franka", "ur5"]:
            euler = actions[:, 3:6]
            quat = euler_to_quat(euler, degrees=False)
            return np.concatenate([actions[:, :3], quat, actions[:, 6:]], axis=1)
        return actions

    def _joint_trick(self, actions: np.ndarray) -> np.ndarray:
        """Joint space limits"""
        if self.task_name == "put_pen_into_pencil_case":
            actions = np.clip(actions, np.array(ALOHA_JOINT_MIN), np.array(ALOHA_JOINT_MAX))
        return actions

    def _eef_trick(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """EEF space processing"""
        if self.robot_type in ["ur5", "franka"]:
            actions[:, 3:6] = unwrap_euler_sequence(actions[:, 3:6])
        
        # Task-specific: interpolation smoothing
        if self.task_name in ['arrange_flowers', 'wipe_the_table', 'open_the_drawer']:
            horizon, dim = actions.shape
            x = np.arange(horizon)
            new_x = np.linspace(0, horizon - 1, horizon * 2)
            actions = np.stack([np.interp(new_x, x, actions[:, i]) for i in range(dim)], axis=1)
        
        return actions

    def _gripper_trick(self, actions: np.ndarray) -> np.ndarray:
        """Unified gripper postprocessing"""
        cfg = self.postprocess_args
        threshold = cfg.get("gripper_threshold", 0.01)
        open_val = cfg.get("gripper_open")  # None = keep original value
        close_val = cfg.get("gripper_close", 0.0)
        
        for idx in self.non_delta_mask:
            if idx < actions.shape[1]:
                if open_val is not None:
                    # Binarization mode
                    actions[:, idx] = np.where(
                        actions[:, idx] < threshold, close_val, open_val
                    )
                else:
                    # Threshold clipping mode
                    actions[:, idx] = np.where(
                        actions[:, idx] < threshold, close_val, actions[:, idx]
                    )
        return actions
