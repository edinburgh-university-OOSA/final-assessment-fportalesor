import sys
from pathlib import Path

def setup_paths():
    root = Path(__file__).parent.parent
    sys.path.extend([str(root), str(root/"src")])

setup_paths()