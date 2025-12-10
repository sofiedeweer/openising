import matplotlib.pyplot as plt
import pathlib

from ising.stages.simulation_stage import Ans

def plot_error_SNR(
    ans: dict[int: Ans],
    save: bool = True,
    save_folder: pathlib.Path = ".",
    fig_name: str = "error_SNR",
) -> None:
    """Plots the relative error between the optimal solution and the computed solution for different SNRs.

    Args:
        logfiles (list[pathlib.Path]): List of all the logfiles for the different solvers.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): The path to the folder in which to save the figure. Defaults to ".".
        figName (str, optional): The name of the figure to save. Defaults to error_SNR.
    """
    SNR_values = ans.keys()
    ans_BER = {solver: [] for solver in ans[SNR_values[0]].config.solvers}
    for snr in SNR_values:
        for solver in ans[snr].config.solvers:
            ans_BER[solver].append(ans[snr].BER[solver])

    plt.figure()
    for solver_name, error in ans_BER.items():
        plt.semilogy(SNR_values, error, label=solver_name)
    plt.xlabel("SNR [dB]")
    plt.xticks(SNR_values)
    plt.ylabel("Bit Error Rate")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_folder / f"{fig_name}.pdf", dpi=600, bbox_inches="tight")
    plt.close()
