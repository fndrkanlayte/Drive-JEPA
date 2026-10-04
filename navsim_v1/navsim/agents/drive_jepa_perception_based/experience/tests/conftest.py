"""Make `navsim_v1` importable so the tests run from the repo root:

    pytest navsim_v1/navsim/agents/drive_jepa_perception_based/experience/tests
"""

import sys
from pathlib import Path

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[5]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
