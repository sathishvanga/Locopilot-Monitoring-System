"""
Crew member helper functions

Utility functions for building and validating crew member dictionaries.
"""

from typing import Dict, Optional


def build_crew_members_dict(
    lp_crew_name: Optional[str] = None,
    lp_crew_id: Optional[str] = None,
    alp_crew_name: Optional[str] = None,
    alp_crew_id: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Build crew members dictionary from individual crew member parameters.
    
    Args:
        lp_crew_name: Loco Pilot crew member name (optional)
        lp_crew_id: Loco Pilot crew member ID (optional)
        alp_crew_name: Assistant Loco Pilot crew member name (optional)
        alp_crew_id: Assistant Loco Pilot crew member ID (optional)
    
    Returns:
        Dictionary mapping role ('LP' or 'ALP') to crew member info dict
        with keys: 'name', 'id', 'role'
    
    Example:
        >>> crew = build_crew_members_dict(
        ...     lp_crew_name="John Doe",
        ...     lp_crew_id="LP-001"
        ... )
        >>> crew
        {'LP': {'name': 'John Doe', 'id': 'LP-001', 'role': 'LP'}}
    """
    crew_members: Dict[str, Dict[str, str]] = {}
    
    # Add LP crew if both name and ID are provided and non-empty
    if lp_crew_name and lp_crew_id:
        lp_name = lp_crew_name.strip()
        lp_id = lp_crew_id.strip()
        if lp_name and lp_id:
            crew_members['LP'] = {
                'name': lp_name,
                'id': lp_id,
                'role': 'LP'
            }
    
    # Add ALP crew if both name and ID are provided and non-empty
    if alp_crew_name and alp_crew_id:
        alp_name = alp_crew_name.strip()
        alp_id = alp_crew_id.strip()
        if alp_name and alp_id:
            crew_members['ALP'] = {
                'name': alp_name,
                'id': alp_id,
                'role': 'ALP'
            }
    
    return crew_members


def get_default_crew_name(crew_members: Dict[str, Dict[str, str]]) -> str:
    """
    Get default crew name from crew members dictionary.
    
    Args:
        crew_members: Dictionary mapping role to crew member info
    
    Returns:
        Default crew name, or "Unknown" if no crew members provided
    """
    if crew_members:
        # Return the first crew member's name
        first_crew = list(crew_members.values())[0]
        return first_crew.get('name', 'Unknown')
    return "Unknown"


def get_default_crew_id(crew_members: Dict[str, Dict[str, str]]) -> str:
    """
    Get default crew ID from crew members dictionary.
    
    Args:
        crew_members: Dictionary mapping role to crew member info
    
    Returns:
        Default crew ID, or "N/A" if no crew members provided
    """
    if crew_members:
        # Return the first crew member's ID
        first_crew = list(crew_members.values())[0]
        return first_crew.get('id', 'N/A')
    return "N/A"

