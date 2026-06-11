# conftest.py — pytest configuration
# Ensures project root is on sys.path for all tests

import sys
import os

# Add project root so 'src' is importable from any test file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
