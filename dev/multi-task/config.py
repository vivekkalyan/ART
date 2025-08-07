from pydantic import BaseModel
from typing import Literal, Optional, List


class TaskTrainConfig(BaseModel):
    """Task-specific training configuration with sensible defaults.

    This configuration can be used for any task and provides all the common
    training parameters. Tasks can override get_train_config() to provide
    task-specific defaults.
    """

    # Core training parameters
    trajectories_per_group: int = 8
    groups_per_step: int = 1
    learning_rate: float = 1e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 1000
    num_epochs: int = 4
    minimum_reward_std_dev: float = 0.0

    # Advanced settings (compatible with art.dev.TrainConfig)
    importance_sampling_level: Literal["token", "sequence"] = "token"
    scale_rewards: bool = True
    allow_training_without_logprobs: bool = False
    advantage_balance: float = 0.0
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None

    # Dataset settings
    training_dataset_seed: Optional[int] = None

    # Metrics configuration
    track_metrics: bool = True  # Whether to track and display task-specific metrics
    tracked_metrics: Optional[List[str]] = (
        None  # Specific metrics to track (None = track all)
    )


class TrainerConfig(BaseModel):
    """Configuration for Tasks training

    This configuration controls how multiple tasks are mixed during training
    and provides global overrides for task-specific settings.
    """

    # Mixing strategy for multi-task training
    mixing_strategy: Literal["sequential", "interleaved", "proportional"] = "sequential"
