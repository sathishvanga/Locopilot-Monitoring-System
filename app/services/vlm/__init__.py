"""VLM verification service — split into focused sub-modules.

The original ``app.services.vlm_verification_service`` is now a 3-line
shim that re-exports :class:`VlmVerificationService` from
:mod:`app.services.vlm.service`. New code should import from this
package directly.
"""
from .service import VlmVerificationService, get_vlm_verification_service

__all__ = ["VlmVerificationService", "get_vlm_verification_service"]
