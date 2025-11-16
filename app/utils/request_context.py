"""
Request Context Management - Thread-safe context storage for request metadata

Provides utilities to store and retrieve request-specific metadata
across the application lifecycle.
"""

import contextvars
from typing import Dict, Any, Optional


# Context variable for storing request metadata
_request_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'request_context',
    default={}
)


def set_request_context(context: Dict[str, Any]) -> None:
    """
    Set the request context for the current request
    
    Args:
        context: Dictionary containing request metadata
                (cookie_id, user_id, method, url, request_id, etc.)
    """
    _request_context.set(context)


def get_request_context() -> Dict[str, Any]:
    """
    Get the current request context
    
    Returns:
        Dict[str, Any]: Request context metadata
    """
    return _request_context.get()


def reset_request_context() -> None:
    """
    Reset the request context (cleanup after request completion)
    """
    _request_context.set({})


def update_request_context(updates: Dict[str, Any]) -> None:
    """
    Update specific fields in the request context
    
    Args:
        updates: Dictionary of fields to update
    """
    context = get_request_context()
    context.update(updates)
    _request_context.set(context)


def get_context_value(key: str, default: Any = "N/A") -> Any:
    """
    Get a specific value from request context
    
    Args:
        key: Context key to retrieve
        default: Default value if key not found
        
    Returns:
        Context value or default
    """
    context = get_request_context()
    return context.get(key, default)

