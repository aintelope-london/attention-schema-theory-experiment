import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers, get_score_dimensions
from aintelope.experiments import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:

    timestamp = str(cfg.timestamp)
    logger.info(f"timestamp: {timestamp}")

    logger.info("Running training with the following configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    score_dimensions = get_score_dimensions(cfg)
    run_experiment(cfg, experiment_name="Nonpipeline", score_dimensions=score_dimensions, is_last_pipeline_cycle=False, i_pipeline_cycle=0)


if __name__ == "__main__":
    register_resolvers()
    aintelope_main()
