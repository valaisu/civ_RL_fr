"""One file per scenario; each defines its env class, PRESET and a reward description."""
from ..env import Config
from . import navigate, scout, skirmish

ENVS = {"navigate": navigate.NavigateEnv, "scout": scout.ScoutEnv, "skirmish": skirmish.SkirmishEnv}
PRESETS = {"navigate": navigate.PRESET, "scout": scout.PRESET, "skirmish": skirmish.PRESET}


def make_config(name, **overrides):
    kw = dict(PRESETS[name])
    kw.update(overrides)
    return Config(**kw)


def make_env(cfg: Config):
    return ENVS[cfg.scenario](cfg)
