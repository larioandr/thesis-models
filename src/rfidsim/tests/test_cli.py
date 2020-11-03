import json
import os
from pathlib import Path

import pytest

from .conftest import ROOT_PATH


def test_rfidsim_info_json(coarse_rt01):
    """ Check info JSON file was created and contains two configs for m=1,2.
    """
    file_path = os.path.join(ROOT_PATH, 'rfidsim_coarse_info.json')
    assert os.path.isfile(file_path), f"file '{file_path}' does not exist"

    # Load JSON
    content = Path(file_path).read_text(encoding='utf-8')
    info = json.loads(content)
    assert len(info) == 2, "expected two simulations in info.json"


# vehicles_file_name = 'rfidsim_coarse_0_vehicles.csv'
# vehicles_file_path = os.path.join(root_dir, vehicles_file_name)
# rounds_file_name = 'rfidsim_coarse_0_rounds.txt'
# rounds_file_path = os.path.join(root_dir, rounds_file_name)
# tags_file_name = 'rfidsim_coarse_0_tags.txt'
# tags_file_path = os.path.join(root_dir, tags_file_name)
# packets_file_name = 'rfidsim_coarse_0_packets.txt'
# packets_file_path = os.path.join()
