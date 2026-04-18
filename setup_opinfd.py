import subprocess
import sys
import os

ENV = "opinfd_env"


def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "cases/poisson_1d"

    # Step 1: Create conda environment (skip if exists)
    result = subprocess.run(
        f"conda env list | grep {ENV}",
        shell=True, capture_output=True
    )
    if result.returncode != 0:
        run(f"conda create -y -n {ENV} python=3.10")
    else:
        print(f"[INFO] Conda env '{ENV}' already exists, skipping creation.")

    # Step 2: Install dependencies
    run(f"conda run -n {ENV} pip install -r requirements.txt")

    # Step 3: Run the case
    run(f"conda run -n {ENV} python -m scripts.run_case {case}")


if __name__ == "__main__":
    main()
