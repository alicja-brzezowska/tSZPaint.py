from dataclasses import dataclass

from hydra.core.config_store import ConfigStore



@dataclass
class YProfileConfig:
    alpha: float
    beta0: float  # direct normalization for beta
    gamma: float
    log10_P0: float  # log10 of absolute P0 normalization


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="y_profile_config", node=YProfileConfig)
