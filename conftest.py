# conftest.py
# Ensures the project root is on the Python path during testing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))