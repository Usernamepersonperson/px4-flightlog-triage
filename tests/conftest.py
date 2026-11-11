import pytest

from flightlog_triage import synth
from flightlog_triage.config import load_config


@pytest.fixture(scope="session")
def clean_flight():
    return synth.make_flight(duration_s=240, seed=5)


@pytest.fixture
def cfg():
    return load_config()
