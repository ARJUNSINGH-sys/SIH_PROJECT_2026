"""Launch script for Eigensight SIH 26053 Perception Engine & Web Dashboard.

Python: 3.12 | GPU / CPU
"""

import sys
from pathlib import Path
import uvicorn

# Ensure project root and src are on python path
root = Path(__file__).parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from src.api.server import app

if __name__ == "__main__":
    print("=" * 65)
    print("EIGENSIGHT — SIH 26053 PERCEPTION ENGINE SERVER (PYTHON 3.12)")
    print("Target: DRDO / iDEX — Adaptive Variable-Resolution 2.5D Lidar Mapping")
    print("=" * 65)
    print("Starting FastAPI server at: http://localhost:8000")
    print("Live Dashboard:             http://localhost:8000/")
    print("REST API Docs:              http://localhost:8000/docs")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
