import subprocess
import sys
import shutil

ENV = "opinfd_env"

def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    if shutil.which("conda") is None:
        print("Install Miniconda first")
        sys.exit(1)

    run(f"conda create -y -n {ENV} python=3.10")

    run(f"conda run -n {ENV} pip install -r requirements.txt")

    run(f"conda run -n {ENV} python scripts/run_case.py cases/poisson_1d")

if __name__ == "__main__":
    main()