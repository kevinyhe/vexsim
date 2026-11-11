"""Keep the slow parts of setting up a world on disk for the next process:
compiled models and where things come to rest.  Off until `use` names a
folder.  An entry is found only by the same key, the same vexsim sources,
the same CAD data and the same MuJoCo version."""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, TypeVar

import mujoco

from .override_cad import CAD_DIR

T = TypeVar("T")
_folder: Optional[Path] = None


def use(folder: Path) -> None:
    """Keep setups in `folder` from now on, in this process."""
    global _folder
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _folder = folder


@lru_cache(maxsize=None)
def _sources() -> str:
    simulator = Path(__file__).resolve().parent
    digest = hashlib.sha256(mujoco.__version__.encode())
    for path in sorted(simulator.rglob("*.py")) + sorted(CAD_DIR.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def kept(kind: str, key: tuple, make: Callable[[], T]) -> T:
    """What `make()` gives for `key`, read back when an earlier process kept it."""
    if _folder is None:
        return make()
    name = hashlib.sha256(repr((_sources(), key)).encode()).hexdigest()[:32]
    path = _folder / f"{kind}-{name}.pickle"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    value = make()
    with tempfile.NamedTemporaryFile(dir=_folder, prefix=f".{kind}-", delete=False) as file:
        file.write(pickle.dumps(value))
    os.replace(file.name, path)
    return value
