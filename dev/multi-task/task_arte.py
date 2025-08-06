import asyncio
from typing import List, Generator
import art
from pydantic import BaseModel
from task import Task
from config import TaskTrainConfig
from art_e.data.types_enron import SyntheticQuery
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.local_email_db import generate_database
from art_e.rollout import rollout


class TaskArtE(Task[SyntheticQuery]):
    def __init__(self, name: str = "ART-E"):
        super().__init__(name)

    def get_default_config(self) -> TaskTrainConfig:
        """ART-E specific default configuration."""
        return TaskTrainConfig(
            # max_turns: int = 10,
            # max_tokens: int = 2048,
            # log_to_openpipe: bool = False,
            # stupid_simple_reward_fn: bool = False,
            # include_qwen3_nothink: bool = False,
            # ruler_judge_model: str | None = None
            # messages_only: bool = False
            # Fork configuration
            # fork_from_model: str | None = None
            # fork_from_project: str | None = None
            # fork_not_after_step: int | None = None
            # # Validation configuration
            # num_validation_runs: int = 1  # Number of times to run each validation entry
            trajectories_per_group=6,
            groups_per_step=8,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=4000,
            num_epochs=4,
            scale_rewards=True,
            importance_sampling_level="token",
            minimum_reward_std_dev=0.0,
        )

    def get_dataset(self, split: str) -> Generator[SyntheticQuery, None, None]:
        """
        Returns a generator of scenarios for the given split.
        Each scenario is a SyntheticQuery object.
        """

        # Load queries from the art_e dataset - yields SyntheticQuery objects directly
        yield from load_synthetic_queries(split=split)

    def pre_train(self):
        generate_database()

    async def run(
        self, model: art.Model, scenario: SyntheticQuery, num_samples: int = 1
    ) -> art.TrajectoryGroup:

        class TaskProjectConfig(BaseModel):
            max_turns: int = 10
            max_tokens: int = 2048
            log_to_openpipe: bool = False
            stupid_simple_reward_fn: bool = False
            include_qwen3_nothink: bool = False
            ruler_judge_model: str | None = None
            messages_only: bool = False

        model.config = TaskProjectConfig()

        rollout_coroutines = [rollout(model, scenario) for _ in range(num_samples)]
        trajectories = await asyncio.gather(*rollout_coroutines)
        return art.TrajectoryGroup(trajectories)


# Example usage
if __name__ == "__main__":
    # Create task instance
    task = TaskArtE()

    # Get a sample scenario from different splits
    for split in ["train", "test"]:
        try:
            dataset = task.get_dataset(split)
            scenario = next(dataset)
            print(f"\n{split.upper()} split:")
            print(scenario)
        except Exception as e:
            print(f"\n{split.upper()} split: Not available or error - {e}")

    task.pre_train()
