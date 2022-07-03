from pettingzoo.test import api_test, seed_test

from aintelope.environments import savanna as sut


def test_pettingzoo_api():
    api_test(sut.env(), num_cycles=1000)


def test_seed():
    seed_test(sut.env, num_cycles=10, test_kept_state=True)
