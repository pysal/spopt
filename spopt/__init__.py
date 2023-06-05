import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import locate, region

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spopt")
