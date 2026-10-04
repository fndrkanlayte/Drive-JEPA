"""Phase-A tooling for 'experience retrieval' research on Drive-JEPA (NAVSIM v1).

This package contains only pure, dependency-light helpers (numpy + shapely)
used by the scripts under ``navsim_v1/scripts/experience/``. It must not import
torch or nuplan so it can be unit-tested without the NAVSIM stack.
"""
