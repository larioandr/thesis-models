import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from rfidsim.main import cli


ROOT_PATH = os.path.join('data', 'results', 'rfidsim')


@pytest.fixture
def load_config():
    runner = CliRunner()

    # Load test config from 'tests/data/config.json'
    pwd = os.path.dirname(__file__)
    config_file_name = os.path.join(pwd, 'data', 'config.json')
    config = Path(config_file_name).read_text()

    with runner.isolated_filesystem():
        yield runner, config


@pytest.fixture
def coarse_rt01(load_config):
    runner, config = load_config
    # Write config to temporary folder
    Path('config.json').write_text(config)
    # This config will cause two simulations to be run with m=1,2
    result = runner.invoke(cli, [
        'simulate', 'coarse',
        '--max-real-time=0.1',  # run a very short simulation
        # Override multivalue config parameters:
        '-s reader.inventory.tari:12.5us',
        '--set-value=vehicle.speed:20',
    ])
    assert result.exit_code == 0, result.stdout
    yield result
