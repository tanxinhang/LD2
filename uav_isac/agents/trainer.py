"""Module alias for the isolated legacy trainer."""

import sys

from uav_isac.legacy import trainer as _implementation


sys.modules[__name__] = _implementation
