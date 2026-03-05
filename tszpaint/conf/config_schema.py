from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from tszpaint.config import INTERPOLATORS_PATH


@dataclass
class YProfileConfig:
    alpha: float
    beta: float
    gamma: float

    @property
    def output_file_path(self):
        return (
            INTERPOLATORS_PATH
            / f"alpha={self.alpha}_beta={self.beta}_gamma={self.gamma}.pkl"
        )


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="y_profile_config", node=YProfileConfig)
