import logging
import sys
from ising.stages import TOP, LOGGER
import os
import yaml

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import numpy as np
from ising import api
import matplotlib.pyplot as plt

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
test_parameters = ["seed"]
problem_type = ["Maxcut"]  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, QKP]
config_path = TOP / "ising/inputs/config/example.yaml"
save_folder = TOP / "ising/outputs"
with config_path.open("r") as f:
    config = yaml.safe_load(f)

parameter_values = {test_parameter: config[test_parameter] for test_parameter in test_parameters}
for test_parameter in test_parameters:
    config[test_parameter] = type(parameter_values[test_parameter][0])(0)

for i, problem in enumerate(problem_type):
    print("===============================")
    LOGGER.info(f"Starting simulation for {problem}...")
    # Set up the answer dictionary
    ans = {
        test_parameter: {value: None for value in parameter_values[test_parameter]}
        for test_parameter in test_parameters
    }

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
    if problem == "MIMO":
        energies_base = []
        for _ in range(config["nb_trials"]):
            energies_base.append(ans_base.MIMO[_].lowest_energy["Multiplicative"])
    else:
        best_found = ans_base.best_found
        energies_base = ans_base.energies["Multiplicative"]

    # Run for all different parameters
    for test_parameter in test_parameters:
        print("===============================")
        LOGGER.info(f"Testing parameter: {test_parameter}...")
        for value in parameter_values[test_parameter]:
            new_config[test_parameter] = value
            file = f"ising/inputs/config/config_{problem}_{test_parameter}_{value}.yaml"
            with (TOP / file).open("w") as f:
                yaml.dump(new_config, f)

            ans[test_parameter][value], debug_info = api.get_hamiltonian_energy(
                problem_type=problem, config_path=file, logging_level=logging_level
            )

        new_config[test_parameter] = 0.0

        # Plot results
        parameter_name = test_parameter.replace("_", " ")
        plt.figure()
        plt.hist(
            energies_base,
            bins=15,
            alpha=0.7,
            edgecolor="black",
            label=f"Base run: best energy = {np.min(energies_base):.2f}, avg energy: {np.mean(energies_base):.2f}",
        )
        all_energies = [energies_base]
        for value in parameter_values[test_parameter]:
            if problem == "MIMO":
                energies = []
                for _ in range(config["nb_trials"]):
                    energies.append(ans[test_parameter][value].MIMO[_].lowest_energy["Multiplicative"])
            else:
                energies = ans[test_parameter][value].energies["Multiplicative"]
            all_energies.append(energies)
            plt.hist(
                energies,
                bins=15,
                alpha=0.7,
                edgecolor="black",
                label=f"{parameter_name} = {value}: best energy = {np.min(energies):.2f}, avg energy = {
                    np.mean(energies):.2f}",
            )

        if best_found is not None:
            if best_found < 0.0:
                plt.axvline(
                0.9 * best_found,
                color="k",
                linestyle="-.",
                label=f"90% Best known: {0.9 * best_found}",
                )
            else:
                plt.axvline(
                    1.1 * best_found,
                    color="k",
                    linestyle="-.",
                    label=f"90% Best known: {1.1 * best_found}",
                )
            plt.axvline(best_found, color="k", linestyle="--", label=f"Best known: {best_found}")

        plt.title(f"Energy distribution for different {parameter_name} values - {problem} problem")
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig_name = (
            f"{problem}_{test_parameter}_energy_distribution_{'_flipping' if config['nb_flipping'] > 1 else ''}.png"
        )
        plt.savefig(save_folder / f"figures/{fig_name}", bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.boxplot(all_energies, xticks=["Base"] + [str(val) for val in parameter_values[test_parameter]])
        plt.xlabel(parameter_name)
        plt.ylabel("Energy")
        if best_found is not None:
            if best_found < 0.0:
                plt.axvline(
                0.9 * best_found,
                color="k",
                linestyle="-.",
                label=f"90% Best known: {0.9 * best_found}",
                )
            else:
                plt.axvline(
                    1.1 * best_found,
                    color="k",
                    linestyle="-.",
                    label=f"90% Best known: {1.1 * best_found}",
                )
            plt.axhline(best_found, color="k", linestyle="--", label=f"Best known: {best_found}")

        plt.title(f"Energy distribution for different {parameter_name} values - {problem} problem")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig_name = f"{problem}_{test_parameter}_energy_box_plot{'_flipping' if (config['nb_flipping'] > 1) else ''}.png"
        plt.savefig(
            save_folder / f"figures/{fig_name}",
            bbox_inches="tight",
        )
        plt.close()
    if problem == "MIMO":
        new_config["nb_runs"] = new_config["nb_trials"]
        new_config["nb_trials"] = 1
