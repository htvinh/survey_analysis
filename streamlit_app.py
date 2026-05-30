# streamlit_app.py (Root of repo)
import sys
import os

# Ensure the 'src' directory is in the path
sys.path.append(os.path.abspath("src"))

# Import and run your app
from src.app import main

if __name__ == "__main__":
    main()
