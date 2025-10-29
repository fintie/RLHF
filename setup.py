# setup.py
import subprocess
import sys

def install_requirements():

    requirements = [
        "torch>=2.0.0,<2.2.0",  
        "transformers>=4.30.0,<4.37.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "trl>=0.7.0,<0.8.0",
        "peft>=0.4.0",
        "wandb>=0.15.0",
        "numpy",
        "matplotlib",
        "tqdm",
        "scipy"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

if __name__ == "__main__":
    install_requirements()
    print("Dependency installation completed!")