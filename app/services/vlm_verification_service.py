"""Back-compat shim: redirects this module to ``app.services.vlm.service``.

The implementation has moved to the :mod:`app.services.vlm` package.
Replacing ``sys.modules[__name__]`` with the new module makes any
``import app.services.vlm_verification_service`` (and any
``monkeypatch.setattr`` against this module) act on the new module
directly, preserving every existing import / patching pattern.
"""
import sys
from app.services.vlm import service as _service

sys.modules[__name__] = _service
