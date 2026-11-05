"""Экспорт слоёв web/layers.yaml в COG: data/web/cogs/ и data/web/manifest.json.

.venv\\Scripts\\python.exe web\\scripts\\export_cogs.py --only dem_elevation --only magnetics_anomaly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from jerqazyna_web.export import main

if __name__ == "__main__":
    sys.exit(main())
