from typing import List, Generator, TypeVar, Generic
import art
from config import TaskTrainConfig

TScenario = TypeVar("TScenario")


class Task(Generic[TScenario]):
    def __init__(self, name: str):
        self.name = name

    def get_default_config(self) -> TaskTrainConfig:
        """
        Returns the default training config for this task.
        Subclasses should override this to provide task-specific defaults.
        """
        return TaskTrainConfig()  # Base defaults

    def get_dataset(self, split: str) -> Generator[TScenario, None, None]:
        """
        Returns a generator of scenarios for the given split (e.g., 'train', 'val', 'test').
        Each scenario contains all data needed to run a rollout.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def pre_train(self):
        """
        Hook that is called before training
        Useful for setting up databases etc.
        """
        pass

    async def run(
        self, model: art.Model, scenario: TScenario, num_samples: int = 1
    ) -> List[art.Trajectory]:
        """
        Run model on scenarios and return trajectories with rewards.

        - Can be single scenario -> single trajectory (just loop internally)
        - Can be batch of scenarios -> batch of trajectories
        - Can evaluate rewards immediately or use LLM judge on the batch
        - Can be single-turn or multi-turn
        """
        raise NotImplementedError
