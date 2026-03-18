import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any


from ising.utils.HDF5Logger import return_metadata
from ising.postprocessing.helper_functions import get_metadata_from_logfiles
from ising.stages.simulation_stage import Ans
from ising.utils.flow import relative_to_best_found


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
    ans_data: dict[str : dict[Any : list[Ans]]],
    parameter_name: str,
    parameter_values: list[Any],
    problems: list[str],
    save_folder: pathlib.Path,
    fig_name: str,
):
    """Plots the pareto curve for a parameter from a solver over different benchmarks.

    Args:
        energy_data (dict[str:dict[Any:list]]): a dictionary containing the energy data
                                                for all the different benchmarks.
        parameter_name (str): the name of the parameter. This will be put on the x-axis.
        parameter_values (list[Any]): the list of all the parameter values tested.
        problems (list[str]): the different benchmarks tested.
        best_found (dict[str:float]): a dictionary with the best found energies for every benchmark.
        save_folder (pathlib.Path): where to save the figure.
        fig_name (str): name of the figure to save.
        solver (str): the solver to plot the pareto curve for.
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:grey"]
    error_colors = ["darkblue", "chocolate", "darkgreen", "maroon"]
    bar_width = 0.4
    x = np.arange(0, (4 * bar_width) * len(parameter_values), bar_width * 4)
    for solver in ans_data[problems[0]][parameter_values[0]][0].config.solvers:
        plt.figure()
        fig, ax = plt.subplots()
        # ax = fig.get_axes()[0]
        ax2 = ax.twinx()
        for ind, problem in enumerate(problems):
            energies_avg = {val: 0.0 for val in parameter_values}
            energies_std = {val: 0.0 for val in parameter_values}
            for val, ans_list in ans_data[problem].items():
                # Store the energies as a relative error to the best found
                energies = np.array([])
                for ans in ans_list:
                    energies = np.append(
                        energies, relative_to_best_found(np.array(ans.energies[solver]), ans.best_found)
                    )

                energies_avg[val] = np.mean(energies)
                energies_std[val] = np.std(energies)
            energies_avg = [energies_avg[val] for val in parameter_values]
            energies_std = [energies_std[val] for val in parameter_values]
            if problem != "MIMO":
                ax.plot(
                    x,
                    energies_avg,
                    color=colors[ind],
                    fmt="o",
                    label=str(problem),
                )
                ax.fill_between(
                    x,
                    energies_avg-energies_std,
                    energies_avg+energies_std,
                    color=error_colors[ind],
                )
            # else:
            #     ax2.errorbar(
            #         x,
            #         energies_avg,
            #         yerr=energies_std,
            #         color=colors[ind],
            #         linestyle="--",
            #         marker="*",
            #         label=str(problem),
            #     )
            #     ax2.set_ylabel("Bit Error Rate", color=colors[ind], fontsize=15)
        ax.set_yscale("log")
        # ax2.set_yscale("log")
        ax.set_ylim(1e-4, 1e5)
        # ax2.set_ylim(1e-4, 1)
        ax.set_xticks(x, [str(val) for val in parameter_values])
        ax.set_xlabel(parameter_name, fontsize=15)
        ax.set_ylabel("Relative distance to best found energy", fontsize=15)
        ax.set_title(f"Pareto curve for different {parameter_name} values - {solver} solver", fontsize=15)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        leg = ax2.legend(handles1 + handles2, labels1 + labels2, fontsize=15, loc="upper left")
        leg.set_zorder(100)
        fig.savefig(save_folder / fig_name, bbox_inches="tight", dpi=600)
        plt.close()
