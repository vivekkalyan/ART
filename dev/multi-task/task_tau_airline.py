import asyncio
from types import SimpleNamespace
from typing import Generator, Optional
import art
from task import Task
from config import TaskTrainConfig
from tau_bench.envs.airline.env import MockAirlineDomainEnv
from tau_bench.envs.user import UserStrategy
from tau_bench.types import TauBenchPolicyConfig, RunConfig
from typing import Tuple
import sys
import os

# Add the tau-bench directory to the path to import from run_rl
tau_bench_path = os.path.join(os.path.dirname(__file__), "..", "tau-bench")
if not os.path.exists(tau_bench_path):
    # Try absolute path if relative doesn't work
    tau_bench_path = "/root/ART/dev/tau-bench"
sys.path.insert(0, tau_bench_path)

try:
    from run_rl import rollout_tau_bench_task
except ImportError:
    # Fallback: try importing from current working directory
    import importlib.util

    run_rl_path = os.path.join(tau_bench_path, "run_rl.py")
    spec = importlib.util.spec_from_file_location("run_rl", run_rl_path)
    run_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_rl)
    rollout_tau_bench_task = run_rl.rollout_tau_bench_task


class TaskTauAirline(Task[Tuple[int, str]]):
    """
    Task wrapper for tau-bench airline domain.

    The tau-bench airline domain simulates customer service interactions
    for an airline company, including flight bookings, cancellations,
    and modifications.
    """

    def __init__(self, name: str = "tau-airline"):
        super().__init__(name)

    def get_train_config(self) -> TaskTrainConfig:
        """Tau-bench airline specific default configuration."""
        return TaskTrainConfig(
            trajectories_per_group=64,
            groups_per_step=32,
            learning_rate=5e-7,
            eval_steps=8,
            val_set_size=60,
            training_dataset_size=32,
            num_epochs=50,
            scale_rewards=True,
            importance_sampling_level="token",
        )

    def get_dataset(self, split: str) -> Generator[Tuple[int, str], None, None]:
        """
        Returns a generator of scenarios for the given split.
        Following run_rl.py convention: train uses first N tasks, test/val uses next M tasks.
        """
        # we can hard code these values for now, since its only used to get the dataset (and it doesnt depend on these
        # values)
        env = MockAirlineDomainEnv(
            user_strategy=UserStrategy.LLM,
            user_model="gpt-4.1",
            user_provider="openai",
            task_split="test",
        )

        # Get default config for dataset sizes
        config = self.get_train_config()

        # Determine which task indices to yield based on the requested split
        if split == "train":
            # Training tasks: first training_dataset_size tasks
            task_indices = range(min(config.training_dataset_size, len(env.tasks)))
        elif split in ["test", "val", "dev"]:
            # Test/validation tasks: next val_set_size tasks after training
            start_idx = config.training_dataset_size
            end_idx = start_idx + config.val_set_size
            task_indices = range(start_idx, min(end_idx, len(env.tasks)))
        else:
            raise ValueError(f"Unknown split: {split}")

        # Yield tasks for the specified indices
        for idx in task_indices:
            if idx < len(env.tasks):
                yield (idx, split)

    async def run(
        self,
        model: art.TrainableModel,
        scenario: Tuple[int, str],
        num_samples: int = 1,
    ) -> art.TrajectoryGroup:
        """
        Run model on airline customer service scenarios and return trajectories with rewards.
        """
        task_index, split = scenario

        model.config = SimpleNamespace()
        model.config.run_config = RunConfig(
            env="airline",
            model_provider="hosted_vllm",
            user_model_provider="openai",
            model=model.name,
            user_model="gpt-4.1",
            user_strategy=UserStrategy.LLM,
            agent_strategy="tool-calling-rl",
            temperature=1.0,
            task_split="test",
            api_key=model.inference_api_key,
            base_url=model.inference_base_url,
            base_model=model.base_model,
        )

        # Generate trajectories using rollout_tau_bench_task
        rollout_coroutines = [
            rollout_tau_bench_task(
                model=model,
                task_index=task_index,
                step=0,
                phase=split,
                reward_type="real",
                is_shadow=False,
            )
            for _ in range(num_samples)
        ]
        trajectories = await asyncio.gather(*rollout_coroutines)
        return art.TrajectoryGroup(trajectories)


# Example usage
if __name__ == "__main__":
    # Create task instance
    task = TaskTauAirline()

    # Get a sample scenario from different splits
    for split in ["train", "test"]:
        try:
            dataset = task.get_dataset(split)
            task_index, task_split = next(dataset)
            print(f"\n{split.upper()} split:")
            print(f"  Task index: {task_index}")
            print(f"  Split: {task_split}")
        except Exception as e:
            print(f"\n{split.upper()} split: Not available or error - {e}")
