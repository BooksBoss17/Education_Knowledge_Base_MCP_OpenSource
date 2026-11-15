"""Public installer entry; models are provided by the user."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2]/"scripts/setup.py"),run_name="__main__")
