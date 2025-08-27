# Overview

The Task class is a flexible interface for defining training tasks in a reinforcement learning or fine-tuning context. It supports various task types including single-turn reasoning, multi-turn interactions, tool use, and different evaluation strategies.

## Task Class Interface

```python
from typing import Generator, TypeVar, Generic
import art
from config import TaskTrainConfig

TScenario = TypeVar("TScenario")

class Task(Generic[TScenario]):
    def __init__(self, name: str):
        self.name = name
    
    def get_train_config(self) -> TaskTrainConfig:
        """Returns task-specific training configuration"""
        return TaskTrainConfig()  # Base defaults
    
    def get_dataset(self, split: str) -> Generator[TScenario, None, None]:
        """Returns a generator of scenarios for the given split"""
        raise NotImplementedError
    
    def pre_train(self):
        """Hook called before training (e.g., for database setup)"""
        pass
    
    async def run(
        self, 
        model: art.TrainableModel, 
        scenario: TScenario,
        num_samples: int = 1
    ) -> art.TrajectoryGroup:
        """Run model on a scenario and return trajectory group with rewards"""
        raise NotImplementedError
```

## Method Specifications

### get_train_config()

**Purpose:** Provide task-specific training configuration defaults

**Returns:** TaskTrainConfig with task-specific settings

**Note:** Override this method to customize training parameters for your task

### get_dataset(split: str)

**Purpose:** Load and yield scenarios for training/validation/testing

**Parameters:**
- `split`: One of "train", "val", or "test"

**Returns:** Generator yielding scenario objects of type TScenario

**Note:** Each scenario should contain all data needed for one run() call

### pre_train()

**Purpose:** Setup hook called before training begins

**Use cases:**
- Initialize databases or external resources
- Precompute data that will be reused
- Set up task-specific infrastructure

### run(model, scenario, num_samples)

**Purpose:** Execute the task and compute rewards

**Parameters:**
- `model`: The TrainableModel being trained/evaluated
- `scenario`: Single scenario from the dataset (type TScenario)
- `num_samples`: How many trajectories to generate (default=1)

**Returns:** art.TrajectoryGroup containing:
- Multiple trajectories with rewards
- Group-level processing capabilities
- Support for group judging and reward normalization

## Implementation Patterns

### Pattern 1: Single-Turn Task

For tasks with one input and one output (e.g., question answering, math problems):

```python
class MathTask(Task[Dict[str, Any]]):
    def __init__(self):
        super().__init__("math_reasoning")
    
    def get_dataset(self, split: str) -> Generator[Dict[str, Any], None, None]:
        problems = load_math_problems(split)
        for problem in problems:
            yield {
                "question": problem.question,
                "answer": problem.answer,
                "problem_id": problem.id
            }
    
    async def run(self, model: art.TrainableModel, scenario: Dict[str, Any], num_samples: int = 1) -> art.TrajectoryGroup:
        trajectories = []
        
        for _ in range(num_samples):
            # Build conversation
            messages = [
                {"role": "system", "content": "Solve the problem step by step."},
                {"role": "user", "content": scenario["question"]}
            ]
            
            # Get model response
            response = await model.generate(messages)
            
            # Create trajectory
            traj = art.Trajectory(
                messages_and_choices=messages + [response],
                reward=0,  # Will be set after evaluation
                metadata={"problem_id": scenario["problem_id"]}
            )
            
            # Evaluate correctness (could use LLM judge or exact match)
            is_correct = await self.check_answer(response["content"], scenario["answer"])
            traj.reward = 1.0 if is_correct else 0.0
            traj.metrics = {"correct": is_correct}
            
            trajectories.append(traj)
        
        return art.TrajectoryGroup(trajectories)
```

### Pattern 2: Multi-Turn Task with Tools

For tasks involving multiple interactions and tool usage:

```python
class EmailSearchTask(Task[Dict[str, Any]]):
    def __init__(self):
        super().__init__("email_search")
        self.max_turns = 10
    
    def get_dataset(self, split: str) -> Generator[Dict[str, Any], None, None]:
        queries = load_email_queries(split)
        for query in queries:
            yield {
                "question": query.question,
                "inbox_address": query.inbox,
                "correct_email_id": query.target_email_id,
                "expected_answer": query.answer
            }
    
    async def run(self, model: art.TrainableModel, scenario: Dict[str, Any], num_samples: int = 1) -> art.TrajectoryGroup:
        # For multi-turn tasks, typically num_samples=1
        # But you could run multiple episodes with different random seeds
        
        trajectories = []
        for _ in range(num_samples):
            traj = await self.run_episode(model, scenario)
            trajectories.append(traj)
        
        return art.TrajectoryGroup(trajectories)
    
    async def run_episode(self, model: art.Model, scenario: Dict[str, Any]) -> art.Trajectory:
        traj = art.Trajectory(
            messages_and_choices=[],
            reward=0,
            metadata=scenario
        )
        
        # Initialize conversation
        traj.messages_and_choices.extend([
            {"role": "system", "content": "You are an email search assistant..."},
            {"role": "user", "content": scenario["question"]}
        ])
        
        # Define tools
        tools = [
            {"name": "search_emails", "parameters": {...}},
            {"name": "read_email", "parameters": {...}},
            {"name": "submit_answer", "parameters": {...}}
        ]
        
        # Run conversation loop
        found_answer = False
        for turn in range(self.max_turns):
            response = await model.generate(
                messages=traj.messages(),
                tools=tools
            )
            traj.messages_and_choices.append(response)
            
            # Process tool calls
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    result = await self.execute_tool(tool_call, scenario)
                    traj.messages_and_choices.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"]
                    })
                    
                    if tool_call["function"]["name"] == "submit_answer":
                        found_answer = True
                        submitted_answer = json.loads(tool_call["function"]["arguments"])["answer"]
                        break
            
            if found_answer:
                break
        
        # Calculate reward
        if found_answer:
            is_correct = submitted_answer == scenario["expected_answer"]
            traj.reward = 1.0 if is_correct else 0.0
        else:
            traj.reward = -0.5  # Penalty for not submitting answer
        
        traj.metrics = {
            "found_answer": found_answer,
            "turns_used": turn + 1,
            "correct": traj.reward > 0
        }
        
        return traj
```

### Pattern 3: Multiple Samples with Comparative Evaluation

For tasks where you want to generate multiple outputs and compare them:

```python
class CreativeWritingTask(Task[Dict[str, Any]]):
    def __init__(self):
        super().__init__("creative_writing")
    
    async def run(self, model: art.TrainableModel, scenario: Dict[str, Any], num_samples: int = 1) -> art.TrajectoryGroup:
        # Generate multiple stories
        trajectories = []
        messages = [
            {"role": "system", "content": "Write a creative story."},
            {"role": "user", "content": f"Write about: {scenario['prompt']}"}
        ]
        
        for i in range(num_samples):
            response = await model.generate(messages, temperature=1.0)
            traj = art.Trajectory(
                messages_and_choices=messages + [response],
                reward=0,
                metadata={"sample_id": i, **scenario}
            )
            trajectories.append(traj)
        
        # Evaluate based on number of samples
        if num_samples == 1:
            # Absolute scoring
            quality_score = await self.judge_quality(trajectories[0])
            trajectories[0].reward = quality_score
        else:
            # Comparative scoring - rank all stories
            rankings = await self.rank_stories(trajectories)
            for traj, rank in zip(trajectories, rankings):
                # Better rank = higher reward
                traj.reward = 1.0 - (rank - 1) / num_samples
        
        return art.TrajectoryGroup(trajectories)
```

### Pattern 4: Immediate Reward Calculation

For tasks where rewards can be computed immediately without external evaluation:

```python
class CodeExecutionTask(Task[Dict[str, Any]]):
    async def run(self, model: art.TrainableModel, scenario: Dict[str, Any], num_samples: int = 1) -> art.TrajectoryGroup:
        trajectories = []
        
        for _ in range(num_samples):
            messages = [
                {"role": "system", "content": "Write Python code to solve the problem."},
                {"role": "user", "content": scenario["problem"]}
            ]
            
            response = await model.generate(messages)
            code = extract_code(response["content"])
            
            # Execute code and check test cases
            passed_tests = 0
            for test in scenario["test_cases"]:
                try:
                    result = execute_code(code, test["input"])
                    if result == test["expected_output"]:
                        passed_tests += 1
                except:
                    pass  # Failed test
            
            reward = passed_tests / len(scenario["test_cases"])
            
            traj = art.Trajectory(
                messages_and_choices=messages + [response],
                reward=reward,
                metrics={
                    "passed_tests": passed_tests,
                    "total_tests": len(scenario["test_cases"])
                }
            )
            trajectories.append(traj)
        
        return art.TrajectoryGroup(trajectories)
```

## Best Practices

1. **Scenario Design:** Include all necessary information in each scenario dict
   ```python
   # Good
   yield {
       "input": "...",
       "expected_output": "...",
       "metadata": {...}  # Additional context
   }
   ```

2. **Error Handling:** Always handle model/API failures gracefully
   ```python
   try:
       response = await model.generate(messages)
   except Exception as e:
       traj.reward = -1.0  # Failure penalty
       traj.metrics = {"error": str(e)}
   ```

3. **Metrics:** Include relevant metrics for analysis
   ```python
   traj.metrics = {
       "primary_metric": value,  # e.g., "correct": True
       "efficiency_metric": value,  # e.g., "turns_used": 3
       "quality_metric": value,  # e.g., "confidence": 0.95
   }
   ```

4. **Memory Efficiency:** For large datasets, use generators properly
   ```python
   def get_dataset(self, split: str):
       # Don't load everything into memory
       with open(f"{split}.jsonl") as f:
           for line in f:
               yield json.loads(line)
   ```

## Usage in Training Loop

```python
# Training loop will use your task like this:
task = YourTask()
dataset = list(task.get_dataset("train"))  # Or iterate directly

for epoch in range(num_epochs):
    for scenario in dataset:
        trajectories = await task.run(
            model, 
            scenario, 
            num_samples=config.get("samples_per_scenario", 1)
        )
        
        # Use trajectories for training
        for traj in trajectories:
            loss = compute_loss(traj)
            optimize(loss)
```

## Summary

The Task framework is designed to be simple yet flexible. The key is that run() takes a single scenario and returns a list of trajectories with computed rewards. Whether those trajectories come from single-turn QA, multi-turn tool use, or comparative evaluation of multiple samples - that's entirely up to your implementation.
