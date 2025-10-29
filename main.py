# main.py
import argparse
import sys
import os

def check_environment():

    try:
        import torch
        import transformers
        import datasets
        import trl
        print("✓ All core dependencies are available")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"TRL version: {trl.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="EfficientRLHF Implementation (Fixed)")
    parser.add_argument("--setup", action="store_true", help="Install dependencies")
    parser.add_argument("--run_demo", action="store_true", help="Run demonstration")
    parser.add_argument("--run_baselines", action="store_true", help="Run baseline experiments")
    parser.add_argument("--run_efficient", action="store_true", help="Run EfficientRLHF experiments")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.setup:
        print("Please run: python setup.py")
        return
    
    if not check_environment():
        print("Please install dependencies first: python setup.py")
        return
    
    if args.test:
        print("Running quick test...")
        from data_loader import RLHFDataLoader
        
        loader = RLHFDataLoader()
        test_data = loader.load_hh_rlhf("train[:5]")
        print(f"✓ Data loading test passed: loaded {len(test_data)} examples")
        return
    
    if args.run_baselines:
        print("Running baseline experiments...")
        from experiment import ExperimentRunner
    
        experiment = ExperimentRunner()
        # 使用正确的方法名
        results = experiment.run_simple_experiment()
        
        print("\nBaseline experiments completed!")
        print("Results:", results)
    
    if args.run_efficient:
        print("Running EfficientRLHF experiments...")
        from efficient_RLHF import run_efficient_rlhf_demo
        
        run_efficient_rlhf_demo()
        print("EfficientRLHF experiments completed!")
    
    if args.run_demo:
        print("Running complete demonstration...")
        from experiment import ExperimentRunner
        
        experiment = ExperimentRunner()
        results = experiment.run_simple_experiment()
        
        print("\nDemonstration completed!")
        print("Results summary:", results)
        
    if args.plot:
        from experiment import ExperimentRunner
        experiment = ExperimentRunner()
        experiment.plot_demo_results()

if __name__ == "__main__":
    main()