import os
import json


def create_results_dir(case_dir):
    results_dir = os.path.join(case_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_metrics(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[OPINFD] Metrics saved -> {path}")
