import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
import pathlib

from ising.stages.simulation_stage import Ans
color = list(colors.BASE_COLORS.values())

def plot_error_SNR(
    ans: dict[int : Ans | dict[str:float]],
    save: bool = True,
    save_folder: pathlib.Path = ".",
    fig_name: str = "error_SNR",
) -> None:
    """Plots the BER for different SNRs.

    Args:
        ans (list[pathlib.Path]): List of all the logfiles for the different solvers.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): The path to the folder in which to save the figure. Defaults to ".".
        figName (str, optional): The name of the figure to save. Defaults to error_SNR.
    """
    SNR_values = list(ans.keys())
    if not isinstance(ans[SNR_values[0]], dict):
        ans_BER = {solver: [] for solver in ans[SNR_values[0]].config.solvers}
        for snr in SNR_values:
            for solver in ans[snr].config.solvers:
                ans_BER[solver].append(ans[snr].BER[solver])
    else:
        ans_BER = {solver: [ans[snr][solver] for snr in SNR_values] for solver in ans[SNR_values[0]].keys()}

    markers = Line2D.markers
    markers = list(markers.keys())

    fig = plt.figure()
    for solver_name, error in ans_BER.items():
        plt.semilogy(
            SNR_values, error, markers[list(ans_BER.keys()).index(solver_name)], linestyle="-", label=solver_name,
            color="k" if solver_name == "ZF" else color[list(ans_BER.keys()).index(solver_name)]
        )
    plt.xlabel("SNR [dB]")
    plt.xticks(SNR_values)
    plt.ylabel("Bit Error Rate")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax = fig.get_axes()[0]
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(0.1*abs((x_right-x_left)/(y_low-y_high)))
    if save:
        plt.savefig(save_folder / f"{fig_name}.pdf", dpi=600, bbox_inches="tight")
    plt.close()
