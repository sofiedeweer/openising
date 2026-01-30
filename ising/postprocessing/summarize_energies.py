import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any


from ising.utils.HDF5Logger import return_metadata
from ising.postprocessing.helper_functions import get_metadata_from_logfiles
from ising.stages.simulation_stage import Ans


def summary_energies(logfiles: list[pathlib.Path], save_dir: pathlib.Path) -> None:
    """Summarizes the energies over multiple sweeps for each solver and benchmark solved.
    The summary will hold the minimum, maximum, average and std values over the sweep.

    Args:
        logfiles (list[pathlib.Path]): a list of all the log files to summarize.
        save_dir (pathlib.Path): where to store the data.
    """
    energies = dict()

    for logfile in logfiles:
        solver_name = return_metadata(logfile, "solver")
        model_name = return_metadata(logfile, "model_name")

        if energies.get((solver_name, model_name)) is None:
            energies[(solver_name, model_name)] = []

        energy = return_metadata(logfile, "solution_energy")
        energies[(solver_name, model_name)].append(energy)

    header = "min max avg std"
    for (solver_name, model_name), all_energies in energies.items():
        summary = np.array([[np.min(all_energies), np.max(all_energies), np.mean(all_energies), np.std(all_energies)]])
        save_path = save_dir / f"{solver_name}_{model_name}_summary.csv"
        np.savetxt(save_path, summary, fmt="%.2f", header=header)


def box_plot_energies_logfiles(
    logfiles: list[pathlib.Path], best_found: float, save_dir: pathlib.Path, discriminate_by: str | None = None
) -> None:
    """Generates a boxplot from the final energy obtained from a list of logfiles.

    Args:
        logfiles (list[pathlib.Path]): the list of logfiles to plot from.
        best_found (float): best found energy to plot as a reference.
        save_dir (pathlib.Path): the save directory.
        discriminate_by (str | None, optional): to discriminate the colors by. Defaults to None.
    """
    data = get_metadata_from_logfiles(
        logfiles, discriminate_by if discriminate_by is not None else "num_iterations", "solution_energy"
    )

    df = []
    for solver_name, info in data.items():
        for x_dat, y_dat in info.items():
            if discriminate_by is not None:
                df.append(pd.DataFrame({"solver": solver_name, discriminate_by: x_dat, "energy": y_dat}))
            else:
                df.append(pd.DataFrame({"solver": solver_name, "energy": y_dat}))
    df = pd.concat(df)

    plt.figure()
    if discriminate_by is not None:
        sns.boxplot(data=df, x="solver", y="energy", hue=discriminate_by)
    else:
        sns.boxplot(data=df, x="solver", y="energy")
    plt.axhline(y=best_found, color="k", linestyle="--", label=f"Best found: {best_found}")
    plt.legend()
    plt.savefig(save_dir / "boxplot_energies.png", bbox_inches="tight")
    plt.close()


def box_plot_energies_loop(
    ans_data: dict[Any:Ans],
    base_ans: Ans,
    parameter_values: list[Any],
    parameter_name: str,
    problem: str,
    best_found: float | None,
    save_folder: pathlib.Path,
    fig_name: str,
):
    """Generates a box plot for different values of `parameter_name`. Per value the box plot are ordened per solver.

    Args:
        ans_data (dict[Any:Ans]): the data of the different value runs.
        base_ans (Ans): the data of the base run (`parameter_name` is turned off).
        parameter_values (list[Any]): list of all the parameter values tested.
        parameter_name (str): the name of the parameter.
        problem (str): the problem being solved.
        best_found (float | None): the best found energy to plot as reference.
        save_folder (pathlib.Path): the folder in which to save the figure.
        fig_name (str): the name of the figure.
    """
    solvers = base_ans.config.solvers
    df_solvers = []

    for solver in solvers:
        df_solvers.append(pd.DataFrame({parameter_name: "Base", "energy": base_ans.energies[solver], "solver": solver}))
        for value in parameter_values:
            if problem == "MIMO":
                energies = []
                nb_trials = ans_data[value].config.nb_trials
                for trials in range(nb_trials):
                    energies.append(ans_data[value].MIMO[trials].lowest_energy[solver])
            else:
                energies = ans_data[value].energies[solver]
            df_solvers.append(pd.DataFrame({parameter_name: str(value), "energy": energies, "solver": solver}))
    df = pd.concat(df_solvers)

    plt.figure()
    sns.boxplot(data=df, x=parameter_name, y="energy", hue="solver")
    if best_found is not None:
        plt.axhline(y=best_found, color="k", linestyle="--", label=f"Best found: {best_found}")
        if best_found < 0.0:
            plt.axhline(
                0.9 * best_found,
                color="k",
                linestyle="-.",
                label=f"90% Best found: {0.9 * best_found}",
            )
        elif best_found > 0.0:
            plt.axhline(
                1.1 * best_found,
                color="k",
                linestyle="-.",
                label=f"90% Best found: {1.1 * best_found}",
            )
    plt.title(f"Energy distribution for different {parameter_name} values - {problem} problem")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel(parameter_name)
    plt.ylabel("Energy")
    plt.savefig(
        save_folder / f"figures/{fig_name}",
        bbox_inches="tight",
    )
    plt.close()


def histogram_energies_loop(
    ans_data: dict[Any:Ans],
    ans_base: Ans,
    parameter_values: list[Any],
    parameter_name: str,
    problem: str,
    best_found: float | None,
    fig_name: str,
    save_folder: pathlib.Path,
):
    """Plots a histogram for all the solvers on the different values of `parameter_name`.

    Args:
        ans_data (dict[Any:Ans]): a dictionary containing the answer data for all the different `parameter_values`.
        ans_base (Ans): the answer data for the base run with `parameter_name` turned off.
        parameter_values (list[Any]): the list of all the parameter values tested.
        parameter_name (str): the name of the parameter.
        problem (str): the problem that was tested.
        best_found (float | None): the best found energy to plot as reference.
        fig_name (str): name of the figure to save.
        save_folder (pathlib.Path): folder where the figure needs to be saved.
    """
    solvers = ans_base.config.solvers
    for solver in solvers:
        if problem == "MIMO":
            energies_base = []
            for trial in range(ans_base.config.nb_trials):
                energies_base.append(ans_base.MIMO[trial].lowest_energy[solver])
        else:
            energies_base = ans_base.energies[solver]
        plt.figure()
        plt.hist(
            energies_base,
            bins=15,
            alpha=0.7,
            edgecolor="black",
            label=f"Base run: best energy = {np.min(energies_base):.2f}, avg energy: {np.mean(energies_base):.2f}",
        )
        for value in parameter_values:
            if problem == "MIMO":
                energies = []
                for trial in range(ans_base.config.nb_trials):
                    energies.append(ans_data[value].MIMO[trial].lowest_energy[solver])
            else:
                energies = ans_data[value].energies[solver]
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
            elif best_found > 0.0:
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
        plt.savefig(save_folder / f"figures/{fig_name}", bbox_inches="tight")
        plt.close()


def pareto_curve_loop(
    ans_data: dict[str : dict[Any:Ans]],
    parameter_name: str,
    parameter_values: list[Any],
    problems: list[str],
    best_found: dict[str:float],
    save_folder: pathlib.Path,
    fig_name: str,
    solver: str,
):
    """Plots the pareto curve for a parameter from a solver over different benchmarks.

    Args:
        ans_data (dict[str:dict[Any:Ans]]): a dictionary containing the answer data for all the different benchmarks.
        parameter_name (str): the name of the parameter. This will be put on the x-axis.
        parameter_values (list[Any]): the list of all the parameter values tested.
        problems (list[str]): the different benchmarks tested.
        best_found (dict[str:float]): a dictionary with the best found energies for every benchmark.
        save_folder (pathlib.Path): where to save the figure.
        fig_name (str): name of the figure to save.
        solver (str): the solver to plot the pareto curve for.
    """
    plt.figure()
    for problem in problems:
        energies_avg = [0.0 for _ in parameter_values]
        energies_std = [0.0 for _ in parameter_values]
        for val, ans in ans_data[problem].items():
            energies_val = []
            if problem == "MIMO":
                for trial in range(ans.config.nb_trials):
                    energies_val.append(ans.MIMO[trial].lowest_energy[solver])
            else:
                energies_val = ans.energies[solver]
            # Store the energies as a relative error to the best found
            energies_val = [
                np.abs(energy - best_found[problem]) / (np.abs(best_found[problem]) if best_found[problem] != 0 else 1)
                for energy in energies_val
            ]
            energies_avg[parameter_values.index(val)] = np.mean(energies_val)
            energies_std[parameter_values.index(val)] = np.std(energies_val)
        plt.errorbar(
            parameter_values,
            energies_avg,
            yerr=energies_std,
            label=f"{problem}",
        )
    plt.yscale("log")
    plt.xlabel(parameter_name)
    plt.ylabel("Relative error to best found energy")
    plt.title(f"Pareto curve for different {parameter_name} values - {solver} solver")
    plt.legend()
    plt.grid(which='both')
    plt.savefig(save_folder / fig_name, bbox_inches="tight")
    plt.close()
