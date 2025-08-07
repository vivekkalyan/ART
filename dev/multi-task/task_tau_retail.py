from typing import Generator, Optional
import art
from task import Task
from tau_bench.envs.retail.env import MockRetailDomainEnv
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


class TaskTauRetail(Task[Tuple[int, dict, str]]):
    """
    Task wrapper for tau-bench retail domain.

    The tau-bench retail domain simulates customer service interactions
    for an e-commerce platform, including order management, returns,
    exchanges, and customer inquiries.
    """

    def __init__(
        self,
        name: str = "tau-retail",
        user_strategy: UserStrategy = UserStrategy.REACT,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = "openai",
    ):
        super().__init__(name)
        self.user_strategy = user_strategy
        self.user_model = user_model
        self.user_provider = user_provider

    def get_dataset(self, split: str) -> Generator[TauTask, None, None]:
    def get_dataset(self, split: str) -> Generator[Tuple[int, dict, str], None, None]:
        """
        Returns a generator of scenarios for the given split.
        Each scenario is a tuple of (task_index, task_info, split) from the retail domain.
        """
        # Create environment to get tasks
        env = MockRetailDomainEnv(
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            task_split=split,
        )

        # Yield each task with its index and split
        for task_index, task in enumerate(env.tasks):
            # Store minimal task info needed for logging
            task_info = {
                "user_id": task.user_id,
                "instruction": task.instruction,
            }
            yield (task_index, task_info, split)

    async def run(
        self, model: art.Model, scenario: Tuple[int, dict, str], num_samples: int = 1
    ) -> art.TrajectoryGroup:
        """
        Run model on retail customer service scenarios and return trajectories with rewards.
        """
        task_index, task_info, split = scenario

        # Create a config for the model if it doesn't have TauBenchPolicyConfig
        if not hasattr(model, "config") or not isinstance(
            model.config, TauBenchPolicyConfig
        ):
            # Create a default config that matches the rollout_tau_bench_task expectations
            run_config = RunConfig(
                model_provider="openai",  # Will be overridden by the model's actual provider
                user_model_provider=self.user_provider or "openai",
                user_model=self.user_model,
                env="retail",  # Important: change to retail
                user_strategy=self.user_strategy.value
                if isinstance(self.user_strategy, UserStrategy)
                else self.user_strategy,
                task_split=split,
                max_num_steps=30,
                reward_type="real",
            )

            # Wrap the model with the config
            model_with_config = type(
                "ModelWithConfig",
                (),
                {
                    "config": TauBenchPolicyConfig(run_config=run_config),
                    "name": getattr(model, "name", "model"),
                    "inference_api_key": getattr(model, "inference_api_key", None),
                    "inference_base_url": getattr(model, "inference_base_url", None),
                    "base_model": getattr(model, "base_model", "gpt-4"),
                    "generate": model.generate,
                    "__getattr__": lambda self, name: getattr(model, name),
                },
            )()
        else:
            model_with_config = model

        # Generate trajectories using rollout_tau_bench_task
        trajectories = []
        for sample_idx in range(num_samples):
            try:
                traj = await rollout_tau_bench_task(
                    model=model_with_config,
                    task_index=task_index,
                    step=0,
                    phase=split,
                    reward_type="real",
                    is_shadow=False,
                )

                # Update metadata to match our format
                traj.metadata["sample_idx"] = sample_idx
                traj.metadata["scenario_user_id"] = task_info["user_id"]

                trajectories.append(traj)

            except Exception as e:
                # If run fails, create a failed trajectory
                failed_traj = art.Trajectory(
                    messages_and_choices=[
                        {"role": "system", "content": "Task failed"},
                        {"role": "assistant", "content": f"Error: {str(e)}"},
                    ],
                    reward=-1.0,
                    metadata={
                        "error": str(e),
                        "scenario_user_id": task_info["user_id"],
                        "sample_idx": sample_idx,
                    },
                    metrics={"failed": True},
                )
                trajectories.append(failed_traj)

        return art.TrajectoryGroup(trajectories)


# Example usage
if __name__ == "__main__":
    # Create task instance
    task = TaskTauRetail()

    # Get a sample scenario from different splits
    for split in ["train", "dev", "test"]:
        try:
            dataset = task.get_dataset(split)
            task_index, task_info, task_split = next(dataset)
            print(f"\n{split.upper()} split:")
            print(f"  Task index: {task_index}")
            print(f"  Split: {task_split}")
            print(f"  Scenario user_id: {task_info['user_id']}")
        except Exception as e:
            print(f"\n{split.upper()} split: Not available or error - {e}")
