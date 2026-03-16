from pathlib import Path
import os

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import pickle
from ising import api
from ising.stages import TOP
import logging
import yaml
from Paper_OpenIsing import BASE_PATH

base_path = BASE_PATH / "MIMO"
fig_path = base_path / "figures"
Path.mkdir(fig_path, parents=True, exist_ok=True)
data_path = base_path / "data"
Path.mkdir(data_path, parents=True, exist_ok=True)
config_path = base_path / "config"
Path.mkdir(config_path, parents=True, exist_ok=True)

problem_type = "MIMO"
config_file = config_path / "config_MIMO_test.yaml"
config_file_change = config_path / "config_MIMO_change.yaml"
with config_file.open("rb") as f:
    config = yaml.safe_load(f)
snr_list = config["dummy_snr"]
config["solvers"] = ["DSA"]
nb_problems = 50

for snr in snr_list:
    config["dummy_snr"] = snr
    for problem in range(nb_problems):
        data_file_zf = data_path / f"{snr}_problem{problem}_ber_zf.pkl"
        config["dummy_seed"] = config["initialization_seed"] + problem
        with config_file_change.open("w") as f:
            yaml.safe_dump(config, f)

        ans, _ = api.get_hamiltonian_energy(
            problem_type=problem_type,
            config_path=str(config_file_change.relative_to(TOP)),
            logging_level=logging.WARNING,
        )

        with data_file_zf.open("wb") as f:
            pickle.dump(ans.BER["ZF"], f, protocol=pickle.HIGHEST_PROTOCOL)
