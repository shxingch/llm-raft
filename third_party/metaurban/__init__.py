from __future__ import annotations

import os

_INNER_DIR = os.path.join(os.path.dirname(__file__), "metaurban")
_INNER_INIT = os.path.join(_INNER_DIR, "__init__.py")

__path__ = [_INNER_DIR]
__file__ = _INNER_INIT

with open(_INNER_INIT, "rb") as _fh:
    exec(compile(_fh.read(), _INNER_INIT, "exec"), globals(), globals())
