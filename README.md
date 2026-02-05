# Dexbotic-RoboChallengeInference

## Project Structure

```
Dexbotic-RoboChallengeInference/
├── configs/                        # Hydra configuration files
│   ├── default.yaml               # Default configuration
│   ├── specialist/                # Specialist task configs (30 tasks)
│   │   ├── put_cup_on_coaster.yaml
│   │   ├── arrange_flowers.yaml
│   │   └── ...
│   └── generalist/                # Generalist task configs (30 tasks)
│       ├── put_cup_on_coaster.yaml
│       ├── arrange_flowers.yaml
│       └── ...
├── policies/                       # Policy implementations
│   ├── __init__.py                # Policy factory
│   ├── base_policy.py             # Abstract base policy
│   ├── dm0_policy.py              # DM0 model policy
│   └── dm0_prog_policy.py         # DM0 progress-aware policy
├── runner/                         # Inference orchestration
│   ├── __init__.py
│   └── inference_runner.py        # Preprocessing, inference, postprocessing
├── robot/                          # Robot communication
│   ├── __init__.py
│   ├── interface_client.py        # HTTP client for robot API
│   └── job_worker.py              # Job execution loop
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── constants.py               # Task metadata, robot configs
│   ├── enums.py                   # Enumerations
│   ├── log.py                     # Logging utilities
│   ├── transforms.py              # Coordinate transformations
│   └── util.py                    # General utilities
├── tests/                          # Test suite
│   └── mock_inference_test.py     # Mock inference testing
├── execute.py                      # Main entry point
├── requirements.txt
└── README.md
```

## User Guide

### 1. Installation

First, install the [dexbotic](https://github.com/dexmal/dexbotic) package by following the instructions in its repository.

Then, install the additional dependencies for this project:

```bash
# Clone the repository
git clone https://github.com/RoboChallenge/RoboChallengeInference.git
cd RoboChallengeInference

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Checkpoints

Download model checkpoints from HuggingFace collections and place them under the `checkpoints/` directory:

| Collection | Description |
|:---|:---|
| [DM0-table30-specialist](https://huggingface.co/collections/Dexmal/dm0-table30-specialist) | 30 task-specific specialist models |
| [DM0-table30-generalist](https://huggingface.co/collections/Dexmal/dm0-table30-generalist) | 4 robot-type generalist models (arx5, franka, ur5, aloha) |

```bash
# Download a specialist model
huggingface-cli download Dexmal/DM0-table30_put_cup_on_coaster --local-dir ./checkpoints/DM0-table30_put_cup_on_coaster

# Download a generalist model
huggingface-cli download Dexmal/DM0-table30_generalist_arx5 --local-dir ./checkpoints/DM0-table30_generalist_arx5
```

The expected directory structure:
```
checkpoints/
├── DM0-table30_arrange_flowers/         # Specialist
├── DM0-table30_put_cup_on_coaster/      # Specialist
├── DM0-table30_generalist_arx5/         # Generalist
├── DM0-table30_generalist_franka/       # Generalist
├── DM0-table30_generalist_ur5/          # Generalist
├── DM0-table30_generalist_aloha/        # Generalist
└── ...
```

Each checkpoint directory contains:
```
DM0-table30_*/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── norm_stats.json
├── tokenizer_config.json
└── ...
```

### 3. Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are in `configs/`.

**Default configuration (`configs/default.yaml`):**
```yaml
# @package _global_
task_name: ???          # Required: task name
checkpoint: ???         # Required: model checkpoint path
policy_type: dm0        # Policy type ('dm0', 'dm0_prog')
action_horizon: 15      # Number of action steps
duration: 0.1           # Duration per action (seconds)
image_size: [728, 728]  # Image resolution
log_dir: "./logs"       # Log directory
user_id: ""             # User ID for online mode
job_collection_id: ""   # Job collection ID
postprocess_args:       # Gripper postprocessing
  gripper_threshold: 0.01
  gripper_open: null
  gripper_close: 0.0
```

**Specialist configuration example (`configs/specialist/put_cup_on_coaster.yaml`):**
```yaml
# @package _global_
defaults:
  - ../default

task_name: put_cup_on_coaster
checkpoint: ./checkpoints/DM0-table30_put_cup_on_coaster
action_horizon: 25
```

**Generalist configuration example (`configs/generalist/put_cup_on_coaster.yaml`):**
```yaml
# @package _global_
defaults:
  - ../default

task_name: put_cup_on_coaster
checkpoint: ./checkpoints/DM0-table30_generalist_arx5
action_horizon: 35
```

> **Note:** Specialist models are fine-tuned for a single task. Generalist models are trained across all tasks for a given robot type. Some specialist tasks use `dm0_prog` policy type for progress-aware inference.

### 4. Submit Evaluation on RoboChallenge Website

1. Log in to [RoboChallenge](https://robochallenge.ai) and submit an evaluation request.
2. Wait for task assignment notification.
3. Ensure your code is running during the assigned period (see [Step 5](#5-running-inference)).
4. View results on the "My Submissions" page.

### 5. Running Inference

**Using specialist model:**
```bash
python execute.py --config-name=specialist/put_cup_on_coaster user_id=YOUR_USER_ID job_collection_id=YOUR_JOB_COLLECTION_ID
```

**Using generalist model:**
```bash
python execute.py --config-name=generalist/put_cup_on_coaster user_id=YOUR_USER_ID job_collection_id=YOUR_JOB_COLLECTION_ID
```

**Command-line overrides:**
```bash
python execute.py --config-name=specialist/put_cup_on_coaster user_id=YOUR_USER_ID job_collection_id=YOUR_JOB_COLLECTION_ID checkpoint=/path/to/ckpt action_horizon=20
```
