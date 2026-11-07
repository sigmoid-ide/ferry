"""Ferry: run your Hunch experiments on a GPU machine and bring every run home."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("sigmoid-ferry")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"
