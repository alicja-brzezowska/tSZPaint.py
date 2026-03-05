from dataclasses import dataclass

from hydra.core.config_store import ConfigStore



@dataclass
class YProfileConfig:
    alpha: float
    beta_mul: float
    gamma: float


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="y_profile_config", node=YProfileConfig)
