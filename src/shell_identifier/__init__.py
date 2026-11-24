"""Shell addresses and private dwellings on cadastral parcels (G-NAF, Geoscape Cadastre and Buildings)."""

__version__ = "0.1.0"
__all__ = ["check_frames", "datasets_config", "load_config", "load_frames", "load_inputs", "parse_address",
           "predict_frames", "prepare_context", "run_pipeline", "score_frames", "verify_frames"]

_LAZY = {
    "load_config": ".config",
    "parse_address": ".parser",
    "run_pipeline": ".pipeline",
    "prepare_context": ".pipeline",
    "datasets_config": ".assets",
    "load_inputs": ".assets",
    "verify_frames": ".api",
    "predict_frames": ".api",
    "score_frames": ".api",
    "check_frames": ".api",
    "load_frames": ".api",
}


def __getattr__(name):  # lazy imports keep `parse_address` free of the heavy dependencies
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(name)
