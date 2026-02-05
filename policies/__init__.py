from .base_policy import BasePolicy
from .dm0_policy import DM0Policy
from .dm0_prog_policy import DM0ProgPolicy


def get_policy(ckpt_path: str, policy_type: str = None, **kwargs) -> BasePolicy:
    """
    Factory function: create Policy instance
    
    Args:
        ckpt_path: checkpoint path
        policy_type: policy type ('dm0', 'dm0_prog', etc.)
        **kwargs: Policy initialization parameters
    """
    common_kwargs = {"ckpt_path": ckpt_path, **kwargs}

    if policy_type:
        policy_type = policy_type.lower()
        if policy_type == "dm0":
            return DM0Policy(**common_kwargs)
        elif policy_type == "dm0_prog":
            return DM0ProgPolicy(**common_kwargs)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    # Default to DM0
    return DM0Policy(**common_kwargs)

