from typing import List, Generator
import art
from task import Task
from art_e.data.types_enron import SyntheticQuery
from art_e.data.query_iterators import load_synthetic_queries
from art_e.rollout import rollout
from art_e.project_types import ProjectPolicyConfig
from art import Trajectory
import asyncio


class TaskArtE(Task[SyntheticQuery]):
    def __init__(self, name: str = "ART-E"):
        super().__init__(name)
        self.config = ProjectPolicyConfig()

    def get_dataset(self, split: str) -> Generator[SyntheticQuery, None, None]:
        """
        Returns a generator of scenarios for the given split.
        Each scenario is a SyntheticQuery object.
        """

        # Load queries from the art_e dataset - yields SyntheticQuery objects directly
        yield from load_synthetic_queries(split=split)

    async def run(
        self, model: art.Model, scenario: SyntheticQuery, num_samples: int = 1
    ) -> List[art.Trajectory]:
        """
        Run model on email search scenarios and return trajectories with rewards.
        Uses the existing rollout function from art_e.
        """
        trajectories = []

        for _ in range(num_samples):
            try:
                traj = await rollout(model, scenario)

                trajectories.append(traj)

            except Exception as e:
                # If rollout fails, create a failed trajectory
                failed_traj = Trajectory(
                    messages_and_choices=[
                        {"role": "system", "content": "Task failed"},
                        {"role": "error", "content": str(e)},
                    ],
                    reward=-2.0,
                    metadata={"error": str(e), "scenario_id": scenario.id},
                    metrics={"failed": True, "error_type": type(e).__name__},
                )
                trajectories.append(failed_traj)

        return trajectories


# Example usage
if __name__ == "__main__":
    # Create task instance
    task = TaskArtE()

    # Get a sample scenario
    dataset = task.get_dataset("test")
    scenario = next(dataset)
    print(scenario)
