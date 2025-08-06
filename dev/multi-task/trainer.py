import art
from art.local import LocalBackend
from art.utils import iterate_dataset
from typing import List, Union, Optional, Dict, Any
import random
from dotenv import load_dotenv
import statistics
import polars as pl

from task import Task
from config import TaskTrainConfig, TrainerConfig

load_dotenv()


class TaskTrainer:
    """Generic trainer that handles both single and multi-task training.

    This trainer provides a unified interface for training models on one or more
    tasks, with support for different mixing strategies and task-specific configurations.
    """

    def __init__(
        self,
        tasks: Union[Task, List[Task]],
        model: art.TrainableModel,
        config: Optional[TrainerConfig] = None,
        task_configs: Optional[Dict[str, TaskTrainConfig]] = None,
    ):
        """Initialize the TaskTrainer and the task configuration.

        Args:
            tasks: Single task or list of tasks to train on
            model: The trainable model to use
            config: Global configuration for training (mixing strategy, etc.)
            task_configs: Optional task-specific config overrides
        """
        # Normalize single task to list
        self.tasks = [tasks] if not isinstance(tasks, list) else tasks
        self.model = model
        self.config = config or TrainerConfig()

        # Build final configs with clear precedence:
        # 1. User-provided task_configs (highest priority)
        # 2. Task's get_default_config() (middle priority)
        # 3. TaskTrainConfig defaults (lowest priority)
        self.task_configs = {}
        for task in self.tasks:
            # Start with task's defaults
            task_config = task.get_default_config()

            # Override with user-provided config if exists
            if task_configs and task.name in task_configs:
                user_config = task_configs[task.name]
                # Merge: user config overrides task defaults
                task_config = task_config.model_copy(
                    update=user_config.model_dump(exclude_unset=True)
                )

            self.task_configs[task.name] = task_config

        print(f"Initialized TaskTrainer with {len(self.tasks)} task(s)")
        for task in self.tasks:
            cfg = self.task_configs[task.name]
            print(
                f"  - {task.name}: lr={cfg.learning_rate}, epochs={cfg.num_epochs}, "
                f"traj/group={cfg.trajectories_per_group}, groups/step={cfg.groups_per_step}"
            )

    async def train(self):
        """Train on single or multiple tasks based on config."""
        if len(self.tasks) == 1:
            # Single task training - simple path
            await self._train_single_task(self.tasks[0])
        else:
            # Multi-task training with mixing strategy
            await self._train_multi_task()

    async def _train_single_task(self, task: Task):
        """Train on a single task."""
        print(f"Starting single-task training on {task.name}")
        config = self.task_configs[task.name]

        task.pre_train()

        with LocalBackend() as backend:
            await self.model.register(backend)

            # Load datasets
            print(f"Loading training data for {task.name}...")
            train_scenarios = list(task.get_dataset("train"))
            if (
                config.training_dataset_size
                and len(train_scenarios) > config.training_dataset_size
            ):
                # Sample subset if specified
                if config.training_dataset_seed is not None:
                    random.seed(config.training_dataset_seed)
                train_scenarios = random.sample(
                    train_scenarios, config.training_dataset_size
                )

            print(f"Loading validation data for {task.name}...")
            val_scenarios = list(task.get_dataset("test"))
            if config.val_set_size and len(val_scenarios) > config.val_set_size:
                val_scenarios = val_scenarios[: config.val_set_size]

            print(f"Training data size: {len(train_scenarios)}")
            print(f"Validation data size: {len(val_scenarios)}")

            # Create dataset iterator
            train_iterator = iterate_dataset(
                train_scenarios,
                groups_per_step=config.groups_per_step,
                num_epochs=config.num_epochs,
                initial_step=await self.model.get_step(),
            )

            # Training loop
            for batch in train_iterator:
                # Evaluation
                if batch.step % config.eval_steps == 0:
                    print(f"\n--- Evaluating at Iteration {batch.step} ---")
                    await self._evaluate_task(task, val_scenarios, batch.step)

                # Generate trajectory groups
                groups = await self._generate_trajectory_groups(
                    task, batch.items, config
                )

                # Skip if no valid groups
                if not groups:
                    print(
                        f"WARNING: No valid trajectory groups at step {batch.step}, skipping"
                    )
                    continue

                # Filter groups by reward std dev if configured
                if config.minimum_reward_std_dev > 0:
                    groups = self._filter_groups_by_std_dev(
                        groups, config.minimum_reward_std_dev, batch.step
                    )
                    if not groups:
                        print(
                            f"WARNING: All groups filtered out at step {batch.step}, skipping"
                        )
                        continue

                # Train on the groups
                await self.model.train(
                    groups,
                    config=art.TrainConfig(learning_rate=config.learning_rate),
                    _config=art.dev.TrainConfig(
                        allow_training_without_logprobs=config.allow_training_without_logprobs,
                        scale_rewards=config.scale_rewards,
                        importance_sampling_level=config.importance_sampling_level,
                        advantage_balance=config.advantage_balance,
                        epsilon=config.epsilon,
                        epsilon_high=config.epsilon_high,
                    ),
                )

            # Final evaluation
            print("\n--- Final Evaluation ---")
            await self._evaluate_task(task, val_scenarios, batch.step)

            print(f"Training finished for {task.name}")

    async def _train_multi_task(self):
        """Train on multiple tasks with mixing strategy."""
        print(
            f"Starting multi-task training with {self.config.mixing_strategy} strategy"
        )

        if self.config.mixing_strategy == "sequential":
            await self._train_sequential()
        else:
            raise ValueError(f"Unknown mixing strategy: {self.config.mixing_strategy}")

    async def _train_sequential(self):
        """Train on each task sequentially."""
        for task in self.tasks:
            print(f"\n=== Training on {task.name} ===")
            await self._train_single_task(task)

    async def _generate_trajectory_groups(
        self, task: Task, scenarios: List[Any], config: TaskTrainConfig
    ) -> List[art.TrajectoryGroup]:
        """Generate trajectory groups for a batch of scenarios."""

        groups = await art.gather_trajectory_groups(
            task.run(self.model, scenario, config.trajectories_per_group)
            for scenario in scenarios
        )

        # Filter out None groups (failed judgments)
        return [g for g in groups if g is not None]

    def _filter_groups_by_std_dev(
        self, groups: List[art.TrajectoryGroup], min_std_dev: float, step: int
    ) -> List[art.TrajectoryGroup]:
        """Filter trajectory groups by reward standard deviation."""
        filtered_groups = []

        for grp_idx, g in enumerate(groups):
            rewards = [t.reward for t in g.trajectories]
            if len(rewards) < 2:
                std_dev = 0.0
            else:
                std_dev = statistics.pstdev(rewards)

            if std_dev < min_std_dev:
                print(
                    f"WARNING: Dropping group {grp_idx} at step {step} "
                    f"(std_dev={std_dev:.4f} < {min_std_dev})"
                )
                continue

            filtered_groups.append(g)

        return filtered_groups

    def _aggregate_metrics(
        self, trajectory_groups: List[art.TrajectoryGroup], config: TaskTrainConfig
    ) -> pl.DataFrame:
        """
        Aggregate metrics from multiple trajectory groups for evaluation using Polars.

        Args:
            trajectory_groups: List of trajectory groups from evaluation
            config: Task configuration containing metrics settings

        Returns:
            Dictionary of aggregated metrics with metric names as keys
        """
        # Collect all trajectories
        all_trajectories = []
        for group in trajectory_groups:
            all_trajectories.extend(group.trajectories)

        metrics_data = []
        for t in all_trajectories:
            row = {**t.metrics, "reward": t.reward}
            metrics_data.append(row)

        # Create polars DataFrame
        metrics_df = pl.DataFrame(metrics_data)

        # Filter columns if specific metrics are requested
        if config.tracked_metrics:
            # Keep reward column and requested metrics
            columns_to_keep = [
                col
                for col in metrics_df.columns
                if col == "reward" or col in config.tracked_metrics
            ]
            if columns_to_keep:
                metrics_df = metrics_df.select(columns_to_keep)

        avg_metrics = metrics_df.select(
            [pl.mean(c).alias(c) for c in metrics_df.columns]
        )
        return avg_metrics

    async def _evaluate_task(self, task: Task, val_scenarios: List[Any], step: int):
        """Evaluate model on a task's validation set."""
        print(f"Evaluating {task.name} at step {step}")

        trajectory_groups = await art.gather_trajectory_groups(
            task.run(self.model, scenario, num_samples=1) for scenario in val_scenarios
        )

        # Filter out None groups (failed evaluations)
        valid_groups = [
            g for g in trajectory_groups if g is not None and g.trajectories
        ]

        # Calculate average reward
        total_reward = sum(g.trajectories[0].reward for g in valid_groups)
        avg_reward = total_reward / len(val_scenarios) if val_scenarios else 0
        print(f"  {task.name} - Average reward: {avg_reward:.4f}")

        # Aggregate and display task-specific metrics
        if valid_groups and self.task_configs[task.name].track_metrics:
            metrics = self._aggregate_metrics(
                valid_groups, self.task_configs[task.name]
            ).row(0, named=True)
            if metrics:
                print(f"  {task.name} - Metrics:")
                for metric_name, metric_value in sorted(metrics.items()):
                    print(f"    {metric_name}: {metric_value:.4f}")
