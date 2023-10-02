import logging
import sys

import hydra
from hydra import compose, initialize  # _config_dir
from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers
from aintelope.training.lightning_trainer import run_experiment

logger = logging.getLogger("aintelope.__main__")


# @hydra.main(version_base=None, config_path="config", config_name=test)
def aintelope_main(config_file: str) -> None:  # cfg: DictConfig,
    initialize(config_path="config", job_name="test")
    cfg = compose(config_name=config_file, overrides=[])
    logger.info("Running training with the following configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    run_experiment(cfg)


if __name__ == "__main__":
    register_resolvers()
    aintelope_main(config_file=sys.argv[1])
