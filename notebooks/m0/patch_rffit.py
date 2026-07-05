"""Minimal patch: add a non-interactive `-I` flag to rffit that calls its OWN
identify_satellite_from_doppler() against the /null pgplot device and exits. This
makes rffit's existing identification usable headless (no X/fonts/keypresses) -- the
same idea as the community's `rffit -B`. We do NOT reimplement any estimation.

This M0 notebook just invokes the maintained patcher in `scripts/patch_rffit.py`."""

import sys

from scripts.patch_rffit import patch_rffit

patch_rffit(sys.argv[1] if len(sys.argv) > 1 else "/opt/strf/rffit.c")
