#!/usr/bin/env python3
"""
Example usage of the TaskTrainer for single and multi-task training.

SkyPilot + RunPod Usage:
    # Single task training
    uv run run.py --task arte --fast

    # Multi-task with custom GPU configuration
    uv run run.py --task arte,tau-retail --gpu-type H200-SXM --num-gpus 2

    # Custom cluster name and model settings
    uv run run.py --task tau-retail --cluster-name my-experiment --model-name custom-model-001
"""

import argparse
import asyncio
import art
import os
import textwrap
import logging
from typing import Dict, Optional

from task_arte import TaskArtE
from task_tau_retail import TaskTauRetail
from task_tau_airline import TaskTauAirline
from task_summary import TaskSummary
from task_temporal import TaskTemporal

from trainer import TaskTrainer
from config import TaskTrainConfig, TrainerConfig

from dotenv import load_dotenv

load_dotenv(override=True)

# Task registry
AVAILABLE_TASKS = {
    "arte": TaskArtE,
    "tau-retail": TaskTauRetail,
    "tau-airline": TaskTauAirline,
    "summary": TaskSummary,
    "temporal": TaskTemporal,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models on single or multiple tasks"
    )

    # SkyPilot options
    parser.add_argument(
        "--cluster-name",
        type=str,
        help="SkyPilot cluster name (required for SkyPilot launches)",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="H100-SXM",
        help="GPU type for SkyPilot",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for SkyPilot",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast launch (skip setup) for SkyPilot",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Comma-separated list of tasks to train on. Available: arte, tau-retail, tau-airline, summary, temporal",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name for the model (required)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model to use",
    )
    parser.add_argument(
        "--project", type=str, default="multi_task_rl", help="Project name for tracking"
    )

    # Training strategy
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["sequential", "interleaved", "proportional"],
        default="sequential",
        help="Mixing strategy for multi-task training",
    )

    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument(
        "--trajectories-per-group", type=int, help="Number of trajectories per group"
    )
    parser.add_argument(
        "--groups-per-step", type=int, help="Number of groups per training step"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run single train/val step per task for quick testing",
    )

    return parser.parse_args()


def launch_skypilot_training(args):
    """Launch training on SkyPilot with RunPod."""
    try:
        import sky
        from dotenv import dotenv_values
        from sky import ClusterStatus
    except ImportError:
        print("ERROR: SkyPilot not installed. Install with: uv add --extra runpod .")
        return 1

    # Build training command for remote execution
    train_args = []
    if args.task:
        train_args.extend(["--task", args.task])
    train_args.extend(["--model-name", args.model_name])
    if args.base_model != "Qwen/Qwen2.5-14B-Instruct":
        train_args.extend(["--base-model", args.base_model])
    if args.project != "multi_task_rl":  # Only if changed from default
        train_args.extend(["--project", args.project])
    if args.strategy != "sequential":  # Only if changed from default
        train_args.extend(["--strategy", args.strategy])
    if args.epochs:
        train_args.extend(["--epochs", str(args.epochs)])
    if args.trajectories_per_group:
        train_args.extend(
            ["--trajectories-per-group", str(args.trajectories_per_group)]
        )
    if args.groups_per_step:
        train_args.extend(["--groups-per-step", str(args.groups_per_step)])
    if args.fast_dev_run:
        train_args.append("--fast-dev-run")

    train_cmd = f"uv run run.py {' '.join(train_args)}"
    print(f"Remote training command: {train_cmd}")

    # Setup script
    setup_script = textwrap.dedent(
        """
        echo 'Setting up multi-task training environment...'
        apt update && apt install -y nvtop htop
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        
        # Install dependencies
        uv remove openpipe-art art-e tau-bench summarizer-rl 2>/dev/null || true
        uv add --editable ~/ART --extra backend --extra plotting
        uv add --editable ~/ART/dev/art-e
        uv add --editable ~/ART/dev/tau-bench
        uv add --editable ~/ART/examples/Summary-RL
        uv add --extra runpod .
        uv sync
        
        # Create symlink to tau-bench for import compatibility
        ln -sf ~/ART/dev/tau-bench ./tau-bench
        """
    )

    run_script = textwrap.dedent(f"""
        # Refresh dependencies and ensure proper setup
        uv remove openpipe-art art-e tau-bench summarizer-rl 2>/dev/null || true
        uv add --editable ~/ART --extra backend --extra plotting
        uv add --editable ~/ART/dev/art-e
        uv add --editable ~/ART/dev/tau-bench
        uv add --editable ~/ART/examples/Summary-RL
        
        # Ensure symlink exists for import compatibility
        ln -sf ~/ART/dev/tau-bench ./tau-bench
        
        # Verify setup
        echo "Python path:"
        python -c "import sys; print('\\n'.join(sys.path))"
        echo "Tau-bench directory contents:"
        ls -la ~/ART/dev/tau-bench/ | head -10
        echo "Summary-RL directory contents:"
        ls -la ~/ART/examples/Summary-RL/ | head -10
        echo "Current directory contents:"
        ls -la .
        
        # Start training
        echo "Starting: {train_cmd}"
        {train_cmd}
    """)

    # Create SkyPilot task
    task = sky.Task(
        name="multi-task-training",
        setup=setup_script,
        run=run_script,
        workdir=".",
        envs=dict(dotenv_values()),
    )

    # Set resources
    gpu_spec = f"{args.gpu_type}:{args.num_gpus}"
    task.set_resources(
        sky.Resources(
            accelerators=gpu_spec,
            cloud=sky.clouds.RunPod(),
            region="US",
        )
    )

    task.set_file_mounts({"~/ART": "../.."})

    # Cluster name
    if not args.cluster_name:
        print("ERROR: --cluster-name is required for SkyPilot launches")
        return 1

    cluster_name = args.cluster_name
    cluster_prefix = os.environ.get("CLUSTER_PREFIX")
    if cluster_prefix:
        cluster_name = f"{cluster_prefix}-{cluster_name}"

    print(f"Launching on cluster: {cluster_name}")

    # Check existing cluster
    cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling active jobs...")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
            fast=args.fast,
        )
    )

    print(f"Job submitted (ID: {job_id}). Streaming logs...")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)

    if exit_code == 0:
        print("🎉 Training completed successfully!")
    else:
        print(f"❌ Training failed with exit code {exit_code}")

    return exit_code


async def run_local_training(args):
    """Run the actual training locally (called both locally and remotely via SkyPilot)."""
    import litellm

    # Suprress debug info for Provider list statements
    litellm.suppress_debug_info = True
    # Set LiteLLM logger to WARNING level to suppress INFO messages
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Parse task names
    task_names = [t.strip() for t in args.task.split(",")]

    # Validate tasks
    for task_name in task_names:
        if task_name not in AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(AVAILABLE_TASKS.keys())}"
            )

    # Create task instances
    tasks = []
    for task_name in task_names:
        task_class = AVAILABLE_TASKS[task_name]
        tasks.append(task_class())

    print(f"Selected tasks: {[t.name for t in tasks]}")

    # Create model
    model = art.TrainableModel(
        name=args.model_name,
        project=args.project,
        base_model=args.base_model,
    )

    # Build task configs with overrides
    task_configs: Optional[Dict[str, TaskTrainConfig]] = None

    if any([args.epochs, args.trajectories_per_group, args.groups_per_step]):
        task_configs = {}
        for task in tasks:
            # Create config with only the specified overrides
            override_dict = {}
            if args.epochs is not None:
                override_dict["num_epochs"] = args.epochs
            if args.trajectories_per_group is not None:
                override_dict["trajectories_per_group"] = args.trajectories_per_group
            if args.groups_per_step is not None:
                override_dict["groups_per_step"] = args.groups_per_step

            if override_dict:
                task_configs[task.name] = TaskTrainConfig(**override_dict)

    # Create global config
    global_config = TrainerConfig(
        mixing_strategy=args.strategy,
        fast_dev_run=args.fast_dev_run,
    )

    # Create trainer
    trainer = TaskTrainer(
        tasks=tasks if len(tasks) > 1 else tasks[0],
        model=model,
        config=global_config,
        task_configs=task_configs,
    )

    # Start training
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    await trainer.train()

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


def main():
    """Entry point - decides between SkyPilot launch or local execution."""
    args = parse_args()

    # Check if we're running in a SkyPilot environment (remote)
    # or if user explicitly wants to run locally for testing
    is_remote = os.environ.get("SKYPILOT_NODE_IPS") is not None
    force_local = os.environ.get("MULTI_TASK_LOCAL") == "1"

    if is_remote or force_local:
        # We're running on the remote SkyPilot node, execute training
        return asyncio.run(run_local_training(args))
    else:
        # We're running locally, launch on SkyPilot
        return launch_skypilot_training(args)


if __name__ == "__main__":
    exit(main())


# Example configurations for different training scenarios
"""
Example 1: Single task with ART-E defaults
=========================================
model = art.TrainableModel(
    name="arte-model-001",
    project="art_e_rl",
    base_model="Qwen/Qwen2.5-14B-Instruct"
)

task = TaskArtE()
trainer = TaskTrainer(tasks=task, model=model)
await trainer.train()


Example 2: Single task with custom config
==========================================
task = TaskTauRetail()
trainer = TaskTrainer(
    tasks=task,
    model=model,
    task_configs={
        "tau-retail": TaskTrainConfig(
            learning_rate=5e-6,
            num_epochs=100,
            trajectories_per_group=8
        )
    }
)
await trainer.train()


Example 3: Multi-task with sequential training
================================================
tasks = [TaskArtE(), TaskTauRetail()]

trainer = TaskTrainer(
    tasks=tasks,
    model=model,
    config=TasksConfig(
        mixing_strategy="sequential",
        eval_all_tasks=True
    )
)
await trainer.train()


"""
