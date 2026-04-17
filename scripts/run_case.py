import yaml
import sys
import os

from opinfd.trainer import train_case

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_case.py <case_folder>")
        sys.exit(1)

    case_dir = sys.argv[1]
    config_path = os.path.join(case_dir, "case.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    err = train_case(config, case_dir)

    if err < config["target_error"]:
        print("[SUCCESS] Target achieved")
    else:
        print("[WARNING] Target not reached")