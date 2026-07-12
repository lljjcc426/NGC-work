from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "neurogolf-2026-work" / "scripts"))
from c_deep_rewrite_library import build
if __name__ == "__main__": build("task094")
