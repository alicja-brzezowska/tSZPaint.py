import hydra

from tszpaint.conf.config_schema import YProfileConfig, register_configs
from tszpaint.y_profile.y_profile_jax import run_y_profile

register_configs()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: YProfileConfig):
    run_y_profile(cfg)


if __name__ == "__main__":
    main()
