"""Locopilot test package.

This file exists to ensure pytest's package discovery resolves
``tests.conftest`` to THIS directory's ``conftest.py``, not the
unrelated ``tests`` package shipped inside the conda environment's
site-packages.
"""
