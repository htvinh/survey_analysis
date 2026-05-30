import sys
import os

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports():
    """Smoke test to ensure all refactored modules can be imported."""
    try:
        from src.helpers import logger
        from src.common import read_model, pre_process_data
        from src.quality import compute_cronbach_alpha, compute_correlation
        from src.regression import conduct_regression_analysis
        from src.sem import conduct_sem_analysis
        from src.app import main
        
        logger.info("All modules imported successfully.")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        print("Verification successful: Core modules are healthy.")
        sys.exit(0)
    else:
        print("Verification failed: Check import paths and dependencies.")
        sys.exit(1)
