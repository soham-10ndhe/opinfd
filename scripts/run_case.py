import yaml
import sys
import os

from opinfd.trainer import train_case

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.run_case <case_folder>")
        sys.exit(1)

    case_dir    = sys.argv[1]
    config_path = os.path.join(case_dir, "case.yaml")

    if not os.path.exists(config_path):
        print(f"[ERROR] case.yaml not found at: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"[OPINFD] Running case: {config.get('case_name', case_dir)}")

    err = train_case(config, case_dir)

    if err < config["target_error"]:
        print(f"[SUCCESS] Target achieved  (err={err:.3e} < {config['target_error']:.3e})")
    else:
        print(f"[WARNING] Target not reached (err={err:.3e} >= {config['target_error']:.3e})")
