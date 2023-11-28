import subprocess

from tests.test_config import constants


def test_training_pipeline_main():
    const = constants()
    ret = subprocess.run(["python", "-m", f"{const.PROJECT}"])
    assert ret.returncode == 0, "Trainer from __main__ caused an error!"


def test_training_pipeline_baseline():
    const = constants()
    ret = subprocess.run(["make", f"{const.BASELINE}"])
    assert ret.returncode == 0, "Trainer baseline caused an error!"


def test_training_pipeline_instinct():
    const = constants()
    ret = subprocess.run(["make", f"{const.INSTINCT}"])
    assert ret.returncode == 0, "Trainer baseline caused an error!"
