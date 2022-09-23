import pytest
import yaml
from yaml.loader import SafeLoader


from aintelope.aintelope.training.simple_eval import run_episode


def test_qagent_in_savanna_zoo_sequential():
    # get the default params from training.lightning.yaml
    # then override with these test params

    # Open the file and load the file
    with open('U../../training/lightning.yaml') as f:
        hparams = yaml.load(f, Loader=SafeLoader)
        print(hparams)
    # TODO: refactor out into test constants? Or leave here? /shrug
    test_params = {
        "agent": "q_agent",
        "env": "savanna-zoo-v2",
        "env_entry_point": None,
        "env_type": "zoo",
        "sequential_env": True,
        "env_params": {
            "NUM_ITERS": 40,  # duration of the game
            "MAP_MIN": 0,
            "MAP_MAX": 20,
            "render_map_max": 20,
            "AMOUNT_AGENTS": 1,  # for now only one agent
            "AMOUNT_GRASS_PATCHES": 2,
        },
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
