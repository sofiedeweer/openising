import logging
import sys

from ising.stages import TOP, LOGGER
import os
import yaml

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import pickle
from Paper_OpenIsing import BASE_PATH
from ising import api
from ising.postprocessing.summarize_energies import pareto_curve_loop

"""
This experiments tests the effect of changing a specific parameter of a solver on different problems.
The base yaml file configuration is loaded and modified for each parameter value. Then the configuration file
is used to run the simulation given the specific parameter value.

After all parameter values and problems are tested, the results are plotted in a pareto curve showcasing the relative
best found Ising energy for every problem.
"""

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
test_parameter = "probability_start"
solvers = ["BRIM"]
problem_type = ["Maxcut", "TSP", "QKP", "MIMO"]  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, QKP]

# Create folders for correct saving
save_folder_base = BASE_PATH / "Pareto_curve_parameter"

fig_folder = save_folder_base / "figures"
fig_folder.mkdir(parents=True, exist_ok=True)
data_folder = save_folder_base / "data"
data_folder.mkdir(parents=True, exist_ok=True)
config_folder = save_folder_base / "config"

# Load the original config file
config_path = config_folder / "test_OPS_BRIM.yaml"

with config_path.open("r") as f:
    config = yaml.safe_load(f)

parameter_values = config[test_parameter]
energies_all = {solver: {problem: dict() for problem in problem_type} for solver in solvers}
best_found = {problem: 0 for problem in problem_type}

for i, problem in enumerate(problem_type):
    print("===============================")
    LOGGER.info(f"Starting simulation for {problem}...")

    # Set up new config file
    new_config = config.copy()
    new_config["benchmark"] = config["benchmark"][i]

    if problem == "MIMO":
        new_config["nb_trials"] = config["nb_runs"]
        new_config["nb_runs"] = 1
        new_config["use_multiprocessing"] = False

    # Run for all different parameters
    print("===============================")
    LOGGER.info(f"Testing parameter: {test_parameter}...")
    energies = {solver: {value: [] for value in parameter_values} for solver in solvers}

    for value in parameter_values:
        if test_parameter == "dtSB":
            for solver in solvers:
                new_config[f"dt{solver}"] = value
        else:
            new_config[test_parameter] = value

        # Make new config file
        config_file = config_folder / f"config_{solvers[0]}_{problem}_{test_parameter}_{value}.yaml"
        with (config_file).open("w") as f:
            yaml.dump(new_config, f)

        data_file = data_folder / f"energies_{solvers[0]}_{problem}_{test_parameter}_{value}.pkl"
        if (data_file).exists():
            for solver in solvers:
                with (data_folder / f"energies_{solver}_{problem}_{test_parameter}_{value}.pkl").open("rb") as f:
                    energies[solver][value] = pickle.load(f)
        else:
            ans, debug_info = api.get_hamiltonian_energy(
                problem_type=problem, config_path=str(config_file.relative_to(TOP)), logging_level=logging_level
            )
            for solver in solvers:
                if problem == "MIMO":
                    energies[solver][value] = ans.ber_of_trials[solver]
                else:
                    energies[solver][value] = ans.energies[solver]
                with (data_folder / f"energies_{solver}_{problem}_{test_parameter}_{value}.pkl").open("wb") as f:
                    pickle.dump(energies[solver][value], f)
            with (data_folder / f"best_found_{problem}.pkl").open("wb") as f:
                if problem == "MIMO":
                    pickle.dump(0.0, f)
                else:
                    pickle.dump(ans.best_found, f)
    for solver in solvers:
        energies_all[solver][problem] = energies[solver]

    if problem == "MIMO":
        best_found[problem] = 0.0
    else:
        with (data_folder / f"best_found_{problem}.pkl").open("rb") as f:
            best_found[problem] = pickle.load(f)

    # Plot results
    parameter_name = test_parameter#.replace("_", " ")
    for solver in solvers:
        if solver in parameter_name or solver[1:] in parameter_name:
            parameter_name = parameter_name.replace(solver, "")
            parameter_name = parameter_name.replace(solver[1:], "")
        if parameter_name[-1] == " ":
            parameter_name = parameter_name[:-1]

    if problem == "MIMO":
        new_config["nb_runs"] = new_config["nb_trials"]
        new_config["nb_trials"] = 1

for solver in solvers:
    pareto_curve_loop(
        energies_all[solver],
        parameter_name,
        parameter_values,
        problem_type,
        best_found,
        fig_folder,
        fig_name=f"Pareto_curve_parameter_{parameter_name}_{solver}.pdf",
        solver=solver,
    )
