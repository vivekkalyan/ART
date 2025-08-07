import asyncio
from typing import Generator
import art
from task import Task
from config import TaskTrainConfig
from summarizer import rollout, SummarizerScenario
from summarizer.load_documents import load_documents, Document


class TaskSummary(Task[Document]):
    def __init__(self, name: str = "Summary"):
        super().__init__(name)
        self._val_documents = None
        self._train_documents = None

    def get_default_config(self) -> TaskTrainConfig:
        """Summary-specific default configuration based on Summary-RL settings."""
        return TaskTrainConfig(
            trajectories_per_group=10,
            groups_per_step=10,
            learning_rate=5e-5,
            eval_steps=30,
            val_set_size=91,
            training_dataset_size=3500,
            num_epochs=1,
            scale_rewards=True,
            importance_sampling_level="token",
            minimum_reward_std_dev=0.0,
            track_metrics=True,
            tracked_metrics=[
                "percent",
                "percent_full",
                "percent_diff",
                "word_count",
                "len",
            ],
        )

    def get_dataset(self, split: str) -> Generator[Document, None, None]:
        """
        Returns a generator of scenarios for the given split.
        Each scenario is a Document object containing text and questions.
        """
        # Load documents only once
        if self._val_documents is None or self._train_documents is None:
            self._val_documents, self._train_documents = load_documents()

        if split == "train":
            yield from self._train_documents
        elif split in ["val", "test"]:
            yield from self._val_documents
        else:
            raise ValueError(f"Unknown split: {split}")

    def pre_train(self):
        """
        Pre-training setup. Load documents to ensure they're cached.
        """
        if self._val_documents is None or self._train_documents is None:
            self._val_documents, self._train_documents = load_documents()

    async def run(
        self, model: art.TrainableModel, scenario: Document, num_samples: int = 1
    ) -> art.TrajectoryGroup:
        """
        Run model on a document and return a trajectory group with rewards.

        This follows the Summary-RL implementation where multiple trajectories
        are generated for each document and judged for quality.
        """
        # Create SummarizerScenario from Document
        summarizer_scenario = SummarizerScenario(
            doc=scenario, step=await model.get_step()
        )

        # Generate multiple rollouts for this document
        rollout_coroutines = [
            rollout(model, summarizer_scenario) for _ in range(num_samples)
        ]
        trajectories = await asyncio.gather(*rollout_coroutines)

        return art.TrajectoryGroup(trajectories)


# Example usage
if __name__ == "__main__":
    # Create task instance
    task = TaskSummary()

    # Get a sample scenario from different splits
    for split in ["train", "test"]:
        try:
            dataset = task.get_dataset(split)
            scenario = next(dataset)
            print(f"\n{split.upper()} split:")
            print(f"Document text (first 200 chars): {scenario.document_text[:200]}...")
            print(f"Number of questions: {len(scenario.questions)}")
        except Exception as e:
            print(f"\n{split.upper()} split: Not available or error - {e}")

    task.pre_train()

