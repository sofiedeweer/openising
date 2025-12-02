import logging
import sys
from ising.stages import TOP, LOGGER
import os
import yaml

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

from ising import api
from ising.postprocessing.summarize_energies import box_plot_energies_loop, histogram_energies_loop

"""
This scripts runs a series of simulations for different values of the same parameter.

For each parameter that will be tested, a list of values should be provided in the .yaml file.

As the output a histogram and box plot are generated.
"""

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
test_parameters = ["mismatch_std"]
problem_type = ["MIMO"]  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, QKP]
config_path = TOP / "ising/inputs/config/example.yaml"
save_folder_base = TOP / "ising/outputs"
with config_path.open("r") as f:
    config = yaml.safe_load(f)

parameter_values = {test_parameter: config[test_parameter] for test_parameter in test_parameters}
for test_parameter in test_parameters:
    config[test_parameter] = type(parameter_values[test_parameter][0])(0)

for i, problem in enumerate(problem_type):
    print("===============================")
    LOGGER.info(f"Starting simulation for {problem}...")
    save_folder = save_folder_base / f"{problem}/figures"
    save_folder.mkdir(parents=True, exist_ok=True)
    # Set up new config file
    file = f"ising/inputs/config/config_{problem}_base.yaml"
    new_config = config.copy()
    new_config["benchmark"] = config["benchmark"][i]

    if problem == "MIMO":
        new_config["nb_trials"] = new_config["nb_runs"]
        new_config["nb_runs"] = 1

    with (TOP / file).open("w") as f:
        yaml.dump(new_config, f)

    # Run base simulation
    print("===============================")
    LOGGER.info("Running base simulation...")
    ans_base, debug_info = api.get_hamiltonian_energy(
        problem_type=problem, config_path=file, logging_level=logging_level
    )

    # Save the energies
    energies_base = dict()
    for solver in config["solvers"]:
        energies_base_solver = []
        if problem == "MIMO":
            for _ in range(config["nb_trials"]):
                energies_base_solver.append(ans_base.MIMO[_].lowest_energy[solver])
        else:
            best_found = ans_base.best_found
            energies_base_solver = ans_base.energies[solver]
        energies_base[solver] = energies_base_solver

    # Run for all different parameters
    for test_parameter in test_parameters:
        print("===============================")
        LOGGER.info(f"Testing parameter: {test_parameter}...")
        ans = {value: None for value in parameter_values[test_parameter]}
        for value in parameter_values[test_parameter]:
            new_config[test_parameter] = value
            file = f"ising/inputs/config/config_{problem}_{test_parameter}_{value}.yaml"
            with (TOP / file).open("w") as f:
                yaml.dump(new_config, f)

            ans[value], debug_info = api.get_hamiltonian_energy(
                problem_type=problem, config_path=file, logging_level=logging_level
            )

        new_config[test_parameter] = 0.0

        # Plot results
        parameter_name = test_parameter.replace("_", " ")
        histogram_energies_loop(
            ans_data=ans,
            base_ans=ans_base,
            parameter_values=parameter_values[test_parameter],
            parameter_name=parameter_name,
            problem=problem,
            best_found=best_found,
            fig_name=f"{problem}_{test_parameter}_energy_histogram{'_flipping' if (config['nb_flipping'] > 1)
                                                                   else ''}.png",
            save_folder=save_folder,
        )
        box_plot_energies_loop(
            ans_data=ans,
            base_ans=ans_base,
            parameter_name=parameter_name,
            parameter_values=parameter_values[test_parameter],
            problem=problem,
            best_found=best_found,
            save_folder=save_folder,
            fig_name=f"{problem}_{test_parameter}_energy_box_plot{
                '_flipping' if (config['nb_flipping'] > 1) else ''
            }.png",
        )
    if problem == "MIMO":
        new_config["nb_runs"] = new_config["nb_trials"]
        new_config["nb_trials"] = 1
