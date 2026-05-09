"""Back-compat shim: redirects this module to ``app.core.detectors.sleep.detector``.

The implementation has moved to the :mod:`app.core.detectors.sleep` package.
Replacing ``sys.modules[__name__]`` with the new module makes any
``import app.core.detectors.sleep_detector`` (and any
``monkeypatch.setattr`` against this module) act on the new module
directly, preserving every existing import / patching pattern.
"""
import sys
from app.core.detectors.sleep import detector as _detector

sys.modules[__name__] = _detector
