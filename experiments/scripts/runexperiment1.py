"""
Experiment 1: Ablation Study
Tests the impact of removing key FAIR-CARE components.
"""
import argparse
import csv
import os
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from faircare.orchestration.pipeline import run_pipeline

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Ablation Study")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    parser.add_argument("--configs", required=True, help="Comma-separated config names")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = args.parse_args()

    datasets = args.datasets.split(',')
    configs = args.configs.split(',')
    
    results = []
    
    for dataset in datasets:
        for config_name in configs:
            print(f"\n{'='*60}")
            print(f"Running: {dataset} with {config_name}")
            print(f"{'='*60}\n")
            
            config_path = f"experiments/configs/{config_name}.yaml"
            output_dir = f"results/exp1/{dataset}_{config_name}"
            
            try:
                # Run pipeline
                metrics = run_pipeline(
                    dataset=dataset.strip(),
                    config_path=config_path,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    seed=args.seed
                )
                
                # Collect results
                result = {
                    'dataset': dataset.strip(),
                    'config': config_name.strip(),
                    'SB': metrics.get('components', {}).get('bronze', 0),
                    'SS': metrics.get('components', {}).get('silver', 0),
                    'SG': metrics.get('components', {}).get('gold', 0),
                    'faircarescore': metrics.get('score', 0),
                    'dpd': metrics.get('fairness', {}).get('statistical_parity_difference', 0),
                    'eod': metrics.get('fairness', {}).get('equal_opportunity_difference', 0),
                    'di': metrics.get('fairness', {}).get('disparate_impact', 0),
                    'utility': metrics.get('utility', {}).get('retention', 0),
                    'privacy_risk': metrics.get('privacy', {}).get('risk', 0),
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error running {dataset} with {config_name}: {e}")
                continue
    
    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
