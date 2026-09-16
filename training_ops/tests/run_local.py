"""Run training-ops tests with inherited kernel network denial."""

import sys
import subprocess
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training_ops.tests.network_guard import install

install()
if sys.argv[1:] == ["--shell"]:
    results = []
    for path in sorted(Path(__file__).parent.glob("test_*.sh")):
        print(f"local shell test: {path.name}", flush=True)
        result = subprocess.run(["bash", str(path)], check=False).returncode
        print(f"local shell result: {path.name} status={result}", flush=True)
        results.append(result)
    raise SystemExit(any(results))
loader = unittest.TestLoader()
suite = loader.loadTestsFromNames(sys.argv[1:]) if len(sys.argv) > 1 else loader.discover(str(Path(__file__).parent), pattern="test_*.py", top_level_dir=str(Path(__file__).resolve().parents[2]))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
