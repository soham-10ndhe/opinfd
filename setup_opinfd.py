
import subprocess
import sys
import os

ENV = "opinfd_env"


def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    # Step 1: Create conda environment
    run(f"conda create -y -n {ENV} python=3.10")

    # Step 2: Install dependencies
    run(f"conda run -n {ENV} pip install -r requirements.txt")

    # Step 3: Run the case (FIXED using -m)
    run(f"conda run -n {ENV} python -m scripts.run_case cases/poisson_1d")


if __name__ == "__main__":
    main()