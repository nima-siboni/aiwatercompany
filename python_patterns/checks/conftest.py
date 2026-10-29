"""Load a drill module fresh for each test: either the PR code or the reference fix."""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True  # keep __pycache__ out of the drill folders

PATTERNS_DIR = Path(__file__).resolve().parent.parent


def load_module(path: Path):
    name = f"drill_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickle look modules up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def m(request):
    """The module under test, parametrised by the `bug` / `fine` decorators in harness.py."""
    kind, drill = request.param
    if kind == "pr":
        path = PATTERNS_DIR / "exercises" / f"{drill}.py"
    else:
        path = PATTERNS_DIR / "answers" / f"{drill}_fixed.py"
    module = load_module(path)
    yield module
    sys.modules.pop(module.__name__, None)
