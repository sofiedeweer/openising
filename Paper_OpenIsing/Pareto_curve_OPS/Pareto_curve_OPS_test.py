import logging
import sys
import os
import yaml

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import pickle
import numpy as np
from ising.stages import TOP
from ising import api
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Paper_OpenIsing import BASE_PATH
from ising.utils.flow import relative_to_best_found, compute_ttt

def change_yticks(ax, problem_type:str, best_found:int, large_graph=True):
    if large_graph:
        yticks = ax.get_yticks()[1:-1]
    else:
        yticks= ax.get_yticks()[:-1]
    diff = np.abs(yticks - best_found)
    idx_closest = np.argmin(diff)
    yticks = np.sort(np.append(np.delete(yticks, idx_closest), best_found))
    ax.set_yticks(yticks)
    if problem_type != "Maxcut":
        lab_ax = [f"{t:.1e}" if t != best_found else str(int(t)) for t in yticks]
    else:
        lab_ax = [f"{t:.2e}" if t != best_found else str(int(t)) for t in yticks]
    ax.set_yticklabels(lab_ax, fontsize=18 if large_graph else 15)
    ax.tick_params(axis="both", which="major", labelsize=18 if large_graph else 15)
    for tl in ax.get_yticklabels():
        if tl._y == best_found:
            tl.set_color("red")

# Initialize the logger
logging_level = logging.WARNING
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_types = ["Maxcut", "TSP", "QKP", "MIMO"]  # Specify the problem type [Maxcut, TSP, ATSP, MIMO] "MIMO",

# Initialize all paths
base_path = BASE_PATH / "Pareto_curve_OPS"
data_path = base_path / "data"
data_path.mkdir(parents=True, exist_ok=True)
config_path = base_path / "config"
fig_path = base_path / "figures"

with (config_path / "pareto_curves_Maxcut.yaml").open("r") as f:
    config = yaml.safe_load(f)

nb_runs = config["nb_runs"]
nb_points = len(config["num_iterations_SA"])
solvers = list(config["solvers"])

best_found_list = []

energies = {
    problem_type: {point: {solver: [] for solver in solvers} for point in range(nb_points)}
    for problem_type in problem_types
}
operation_counts = {
    problem_type: {point: {solver: (0, 0) for solver in solvers} for point in range(nb_points)}
    for problem_type in problem_types
}
computation_times = {
    problem_type: {point: {solver: [] for solver in solvers} for point in range(nb_points)}
    for problem_type in problem_types
}
for problem_type in problem_types:
    config_path_problem = config_path / f"pareto_curves_{problem_type}.yaml"
    with config_path_problem.open("r") as f:
        new_config = yaml.safe_load(f)
    num_it_solvers = {
        solver: list(
            new_config[f"num_iterations_{solver}" if (solver != "bSB" and solver != "dSB") else "num_iterations_SB"]
        )
        for solver in solvers
    }
    for point in range(nb_points):
        data_file_dummy = data_path / f"{problem_type}_{solvers[0]}_point{point}_energies.pkl"
        if data_file_dummy.exists():
            for solver in solvers:
                data_file = data_path / f"{problem_type}_{solver}_point{point}_energies.pkl"
                with data_file.open("rb") as f:
                    energies[problem_type][point][solver] = pickle.load(f)

                data_file = data_path / f"{problem_type}_{solver}_point{point}_ops.pkl"
                with data_file.open("rb") as f:
                    operation_counts[problem_type][point][solver] = pickle.load(f)

                data_file = data_path / f"{problem_type}_{solver}_point{point}_time.pkl"
                with data_file.open("rb") as f:
                    computation_times[problem_type][point][solver] = pickle.load(f)
        else:
            for solver in solvers:
                new_config[
                    f"num_iterations_{solver}" if (solver != "bSB" and solver != "dSB") else "num_iterations_SB"
                ] = num_it_solvers[solver][point]
            file = config_path / f"config_pareto_curve_{problem_type}_point{point}.yaml"
            with file.open("w") as f:
                yaml.dump(new_config, f)
            ans, debug_info = api.get_hamiltonian_energy(
                problem_type=problem_type,
                config_path=str(file.relative_to(TOP)),
                logging_level=logging_level,
            )
            if problem_type == "MIMO":
                ans = ans.MIMO[0]
            for solver in solvers:
                energies[problem_type][point][solver] = ans.energies[solver]
                operation_counts[problem_type][point][solver] = (
                    ans.operation_count[solver],
                    ans.operation_count[solver] / num_it_solvers[solver][point],
                )
                computation_times[problem_type][point][solver] = ans.computation_time[solver]

                data_file = data_path / f"{problem_type}_{solver}_point{point}_energies.pkl"
                with data_file.open("wb") as f:
                    pickle.dump(energies[problem_type][point][solver], f)

                data_file = data_path / f"{problem_type}_{solver}_point{point}_ops.pkl"
                with data_file.open("wb") as f:
                    pickle.dump(operation_counts[problem_type][point][solver], f)

                data_file = data_path / f"{problem_type}_{solver}_point{point}_time.pkl"
                with data_file.open("wb") as f:
                    pickle.dump(computation_times[problem_type][point][solver], f)

                best_found_file = data_path / f"{problem_type}_best_found.pkl"
                with best_found_file.open("wb") as f:
                    pickle.dump(ans.best_found, f)
# end if
    if (data_path / f"{problem_type}_best_found.pkl").exists():
        with (data_path / f"{problem_type}_best_found.pkl").open("rb") as f:
            best_found_list.append(pickle.load(f))
    else:
        best_found_list.append(ans[problem_type][0].best_found)

markers = ["s", "d", "^", "o", "p", "v"]

fig_op, axes_op = plt.subplots(1, len(problem_types), figsize=(21, 5))
fig_time, axes_time = plt.subplots(1, len(problem_types), figsize=(21, 5))
labels = dict()
for problem_type, best_found, ax_op, ax_time in zip(problem_types, best_found_list, axes_op, axes_time):
    output_file = base_path / f"{problem_type}_simulation_summary.txt"
    if problem_type != "Maxcut":
        axins_op = inset_axes(ax_op, width="45%", height="65%", loc="upper right")
        axins_time = inset_axes(ax_time, width="45%", height="65%", loc="upper left")
    for solver in solvers:
        mean_en = []
        min_en = []
        max_en = []
        ops = []
        time = []
        with output_file.open("a") as f:
            f.write(f"======= Simulation summary: {solver} =======\n")
        for point in range(nb_points):
            en = energies[problem_type][point][solver]
            ops.append(operation_counts[problem_type][point][solver][0])
            time.append(np.mean(computation_times[problem_type][point][solver]))
            mean_en.append(np.mean(en))
            min_en.append(np.min(en))
            max_en.append(np.max(en))

            relative_error = relative_to_best_found(np.array(en), best_found)
            ttt = compute_ttt(
                np.array(en), np.mean(computation_times[problem_type][point][solver]), best_found, nb_runs
            )
            with output_file.open("a") as f:
                f.write(f"----- Simulation point {point} -----\n")
                f.write(f"Relative error | {np.mean(relative_error):.2e}\n")
                f.write(f"TTT(0.9) | {ttt:.2e}\n")
                f.write(f"OP / it | {operation_counts[problem_type][point][solver][1]:f}\n")
        lab = solver + int(solver == "BRIM") * " (estimated)"
        err = ax_op.errorbar(
            ops,
            mean_en,
            yerr=np.array([np.array(mean_en) - np.array(min_en), np.array(max_en) - np.array(mean_en)]),
            linestyle="-" if solver != "BRIM" else "--",
            marker=markers[solvers.index(solver)],
            alpha=1 if solver != "BRIM" else 0.7,
            label=lab,
            markersize=10
        )
        err = ax_time.errorbar(
            time,
            mean_en,
            yerr=np.array([np.array(mean_en) - np.array(min_en), np.array(max_en) - np.array(mean_en)]),
            linestyle="-" if solver != "BRIM" else "--",
            marker=markers[solvers.index(solver)],
            alpha=1 if solver != "BRIM" else 0.7,
            label=lab,
            markersize=10
        )
        if len(err) > 1:
            labels[lab] = err[0]
        else:
            labels[lab] = err
        if problem_type != "Maxcut":
            axins_op.errorbar(
                ops,
                mean_en,
                yerr=np.array([np.array(mean_en) - np.array(min_en), np.array(max_en) - np.array(mean_en)]),
                linestyle="-" if solver != "BRIM" else "--",
                marker=markers[solvers.index(solver) % len(markers)],
                alpha=1 if solver != "BRIM" else 0.7,
                markersize=9,
            )
            axins_time.errorbar(
                time,
                mean_en,
                yerr=np.array([np.array(mean_en) - np.array(min_en), np.array(max_en) - np.array(mean_en)]),
                linestyle="-" if solver != "BRIM" else "--",
                marker=markers[solvers.index(solver) % len(markers)],
                alpha=1 if solver != "BRIM" else 0.7,
                markersize=9,
            )
    ax_op.axhline(y=best_found, linestyle="--", color="r", label=f"Best found Ising energy: {best_found}")
    ax_op.axhline(
        y=best_found + 0.1 * np.abs(best_found), linestyle="-.", color="k", label="0.9 of best found Ising Energy"
    )
    ax_time.axhline(y=best_found, linestyle="--", color="r", label=f"Best found Ising energy: {best_found}")
    ax_time.axhline(
        y=best_found + 0.1 * np.abs(best_found), linestyle="-.", color="k", label="0.9 of best found Ising Energy"
    )
    ax_op.set_xscale("log")
    ax_op.grid(which="major", linestyle="--")
    ax_op.set_xlabel("Operation count", fontsize=20)

    ax_time.set_xscale("log")
    ax_time.grid(which="major", linestyle="--")
    ax_time.set_xlabel("Computation time [s]", fontsize=20)
    change_yticks(ax_op, problem_type, best_found, True)
    change_yticks(ax_time, problem_type, best_found, True)

    if problem_type == "Maxcut":
        problem_type_ = "Max Cut"
        ax_op.set_ylabel("Ising Energy", fontsize=20)
        ax_time.set_ylabel("Ising Energy", fontsize=20)
    else:
        problem_type_ = problem_type
        axins_op.grid(which="major", linestyle="--")
        axins_op.axhline(y=best_found, linestyle="--", color="r")
        axins_op.axhline(
            y=best_found + 0.1 * np.abs(best_found), linestyle="-.", color="k"
        )
        axins_op.set_xscale("log")
        axins_op.set_xlim((1e7, 3e11))
        if problem_type == "QKP":
            axins_op.set_ylim(best_found - 800, best_found + 6 * np.abs(best_found))
        else:
            axins_op.set_ylim(best_found - 200, best_found + 2 * np.abs(best_found))

        axins_time.grid(which="major", linestyle="--")
        axins_time.axhline(y=best_found, linestyle="--", color="r")
        axins_time.axhline(
            y=best_found + 0.1 * np.abs(best_found), linestyle="-.", color="k"
        )
        axins_time.set_xscale("log")
        axins_time.set_xlim((5e-6, 10))
        if problem_type == "QKP":
            axins_time.set_ylim(best_found - 800, best_found + 6 * np.abs(best_found))
        else:
            axins_time.set_ylim(best_found - 200, best_found + 2 * np.abs(best_found))
        axins_op.yaxis.tick_right()
        change_yticks(axins_op, problem_type, best_found, False)
        axins_time.yaxis.tick_right()
        change_yticks(axins_time, problem_type, best_found, False)
label_best = Line2D([0], [0], color="red", linestyle="--")
label_09 = Line2D([0], [0], color="k", linestyle="-.")

fig_op.tight_layout()
fig_op.legend(
    list(labels.values()) + [label_best, label_09],
    list(labels.keys()) + ["Best found Ising energy", "0.9 of best found Ising energy"],
    fontsize=20,
    loc="upper center",
    ncol=len(solvers)+2,
    fancybox=True,
    bbox_to_anchor=(0.5, 1.1),
    columnspacing=0.3
)
fig_op.savefig(fig_path / "pareto_curve.pdf", bbox_inches="tight")

fig_time.tight_layout()
fig_time.legend(
    list(labels.values()) + [label_best, label_09],
    list(labels.keys()) + ["Best found Ising energy", "0.9 of best found Ising energy"],
    fontsize=20,
    loc="upper center",
    ncol=len(solvers)+2,
    fancybox=True,
    bbox_to_anchor=(0.5, 1.1),
    columnspacing=0.3
)
fig_time.savefig(fig_path / "pareto_curve_time.pdf", bbox_inches="tight")
plt.close()
