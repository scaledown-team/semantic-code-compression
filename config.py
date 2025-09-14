# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY")
if not SCALEDOWN_API_KEY:
    print("Warning: SCALEDOWN_API_KEY not found in .env file.")

HEADERS = {
    "x-api-key": SCALEDOWN_API_KEY,
    "Content-Type": "application/json",
}

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Directory containing the code files to be processed
EXAMPLES_DIR = BASE_DIR / "example_programs"

IGNORE_DIRECTORIES = {".venv", ".git", "__pycache__", ".vscode"}

ALLOWED_EXTENSIONS = {".py"}


# Directory to save output plots
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist

# The proportion of identifiers to mask in the target file. 0.99 means 99%.
MASKING_RATIO = 0.99