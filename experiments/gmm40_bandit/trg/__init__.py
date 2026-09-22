"""Pinned actor-only imports; RL algorithm loaded only when requested."""

def __getattr__(name):
    if name == "OptiQDIME":
        from .algorithm import OptiQDIME
        return OptiQDIME
    raise AttributeError(name)
