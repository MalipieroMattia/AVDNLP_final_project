import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_experiment(config_path):
    """
    Run a single training experiment.

    Args:
        config_path: Path to YAML config file

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"Starting run: {config_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    start_time = datetime.now()

    try:
        # run main.py with the config
        result = subprocess.run(
            [sys.executable, "main.py", "--config", config_path], check=True
        )

        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{'=' * 80}")
        print(f"Completed: {config_path}")
        print(f"Duration: {duration:.2f}s ({duration / 60:.1f} min)")
        print(f"{'=' * 80}\n")

        return True

    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{'=' * 80}")
        print(f"Failed: {config_path}")
        print(f"Duration: {duration:.2f}s ({duration / 60:.1f} min)")
        print(f"Error: {e}")
        print(f"{'=' * 80}\n")

        return False


def main():
    """Run all experiments sequentially."""
    # stanford SNLI experiments
    stanford_configs = [
        "configs/Stanford/bert_lora.yaml",
        "configs/Stanford/bert_partial.yaml",
        "configs/Stanford/distilbert_lora.yaml",
        "configs/Stanford/distilbert_partial.yaml",
    ]

    # Gold commodity news experiments (4 configs)
    # Uncomment these when ready to run Gold experiments
    gold_configs = [
        # "configs/Gold/bert_lora.yaml",
        # "configs/Gold/bert_partial.yaml",
        # "configs/Gold/distilbert_lora.yaml",
        # "configs/Gold/distilbert_partial.yaml",
    ]

    # Combine all configs
    configs = stanford_configs + gold_configs

    # ========================================================================
    # RUN EXPERIMENTS
    # ========================================================================

    print("\n" + "=" * 80)
    print("BATCH EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Total experiments: {len(configs)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Track results
    results = {}
    start_time = datetime.now()

    # run each experiment
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 80}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'=' * 80}")

        success = run_experiment(config)
        results[config] = "Success" if success else "Failed"

    total_duration = (datetime.now() - start_time).total_seconds()
    success_count = sum(1 for status in results.values() if status == "Success")

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for config, status in results.items():
        print(f"[{status}] {config}")

    print(f"\n{'=' * 80}")
    print(f"Completed: {success_count}/{len(configs)} successful")
    print(f"Total time: {total_duration:.2f}s ({total_duration / 60:.1f} min)")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user.\n")
        sys.exit(1)
