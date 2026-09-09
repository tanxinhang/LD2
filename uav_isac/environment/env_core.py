"""Module alias for the isolated legacy environment core."""

import sys

from uav_isac.legacy import environment_core as _implementation


sys.modules[__name__] = _implementation
