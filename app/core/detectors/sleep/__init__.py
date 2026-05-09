"""Sleep detector — split into focused sub-modules.

The original ``app.core.detectors.sleep_detector`` is now a 3-line
shim that re-exports :class:`SleepDetector` from
:mod:`app.core.detectors.sleep.detector`. New code should import from
this package directly.
"""
from .detector import SleepDetector

__all__ = ["SleepDetector"]
