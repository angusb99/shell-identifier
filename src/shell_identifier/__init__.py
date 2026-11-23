"""Shell address identification for G-NAF addresses on Geoscape cadastral parcels."""

__version__ = "0.1.0"
__all__ = ["load_config", "parse_address", "run_pipeline"]


def __getattr__(name):  # lazy imports keep `parse_address` free of geopandas
    if name == "load_config":
        from .config import load_config
        return load_config
    if name == "parse_address":
        from .parser import parse_address
        return parse_address
    if name == "run_pipeline":
        from .pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(name)
