import asyncio
from typing import Generator, TypedDict, Dict
import json
import random
import re
import os
import art
from task import Task
from config import TaskTrainConfig


class TemporalCluePuzzle(TypedDict):
    num_clues: int
    prompt: str
    solution: Dict[str, str]


class TaskTemporal(Task[TemporalCluePuzzle]):
    """
    Task wrapper for temporal clue puzzles.

    The temporal clue task involves solving logical reasoning puzzles
    where the model must deduce the correct order or timing of events
    based on temporal clues provided in natural language.
    """

    def __init__(self, name: str = "temporal-clue"):
        super().__init__(name)
        # Load puzzles from the data file
        # Try multiple possible paths to handle both local and remote environments
        possible_paths = [
            # Remote SkyPilot path
            os.path.expanduser("~/ART/examples/data/temporal-clue/puzzles.json"),
            # Local development path
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "examples",
                "data",
                "temporal-clue",
                "puzzles.json",
            ),
        ]

        puzzles_path = None
        for path in possible_paths:
            if os.path.exists(path):
                puzzles_path = path
                break

        if puzzles_path is None:
            raise FileNotFoundError(
                f"Puzzles data not found. Tried paths: {possible_paths}"
            )

        with open(puzzles_path, "r") as f:
            puzzles: list[TemporalCluePuzzle] = json.load(f)

        # Split into train/val/test following the same pattern as temporal-clue.py
        self.val_puzzles = puzzles[:64]
        self.test_puzzles = puzzles[64:128]
        self.train_puzzles = puzzles[128:]

        # Shuffle train puzzles with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(self.train_puzzles)

    def get_train_config(self) -> TaskTrainConfig:
        """Temporal clue specific default configuration."""
        return TaskTrainConfig(
            trajectories_per_group=50,
            groups_per_step=32,
            learning_rate=6e-6,
            eval_steps=25,
            val_set_size=64,
            training_dataset_size=len(self.train_puzzles),
            num_epochs=1,
            scale_rewards=True,
            importance_sampling_level="token",
        )

    def get_dataset(self, split: str) -> Generator[TemporalCluePuzzle, None, None]:
        """
        Returns a generator of scenarios for the given split.
        Each scenario is a TemporalCluePuzzle object.
        """
        if split == "train":
            yield from self.train_puzzles
        elif split in ["val", "validation"]:
            yield from self.val_puzzles
        elif split == "test":
            yield from self.test_puzzles
        else:
            raise ValueError(f"Unknown split: {split}")

    async def run(
        self, model: art.TrainableModel, scenario: TemporalCluePuzzle, num_samples: int = 1
    ) -> art.TrajectoryGroup:
        """
        Run model on temporal clue puzzles and return trajectories with rewards.

        The reward is calculated as the accuracy - fraction of correctly answered clues.
        """

        async def rollout(puzzle: TemporalCluePuzzle) -> art.Trajectory:
            """Generate a single trajectory for a temporal clue puzzle."""
            messages: art.Messages = [{"role": "user", "content": puzzle["prompt"]}]

            client = model.openai_client()
            chat_completion = await client.chat.completions.create(
                messages=messages, model=model.name
            )

            choice = chat_completion.choices[0]
            content = choice.message.content
            assert isinstance(content, str)

            # Calculate reward based on correct answers
            num_correct = 0
            total_clues = len(puzzle["solution"])

            for key, expected_value in puzzle["solution"].items():
                # Extract answers using the same regex pattern as temporal-clue.py
                matches = re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content)
                if matches:
                    # Take the last match in case of multiple occurrences
                    actual_answer = matches[-1].strip().lower()
                    expected_answer = expected_value.lower()

                    if actual_answer == expected_answer:
                        num_correct += 1

            # Reward is accuracy (0.0 to 1.0)
            reward = accuracy = num_correct / total_clues if total_clues > 0 else 0.0

            return art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=reward,
                metrics={
                    "accuracy": accuracy,
                    "num_correct": num_correct,
                    "total_clues": total_clues,
                    "num_clues": puzzle["num_clues"],
                },
            )

        # Generate trajectories in parallel
        rollout_coroutines = [rollout(scenario) for _ in range(num_samples)]
        trajectories = await asyncio.gather(*rollout_coroutines)

        return art.TrajectoryGroup(trajectories)


# Example usage
if __name__ == "__main__":
    # Create task instance
    task = TaskTemporal()

    # Get a sample scenario from different splits
    for split in ["train", "val", "test"]:
        try:
            dataset = task.get_dataset(split)
            scenario = next(dataset)
            print(f"\n{split.upper()} split:")
            print(f"  Number of clues: {scenario['num_clues']}")
            print(f"  Solution keys: {list(scenario['solution'].keys())}")
            print(f"  Prompt preview: {scenario['prompt'][:100]}...")
        except Exception as e:
            print(f"\n{split.upper()} split: Not available or error - {e}")
