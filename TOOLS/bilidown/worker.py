from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bilidown.backend import execute
if __name__=="__main__":
    execute(Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4:] or None)
