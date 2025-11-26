import pathlib
import numpy as np
from argparse import Namespace
import scipy.sparse.linalg as spalg

from ising.utils.HDF5Logger import return_metadata

from ising.stages.model.ising import IsingModel
from ising.utils.helper_functions import return_rx
from ising.utils.numpy import triu_to_symm


def parse_hyperparameters(args: Namespace) -> dict[str:]:
    """Parses the arguments needed for the solvers.

    Args:
        args (dict): the command line arguments.
        num_iter (int): amount of iterations

    Returns:
        dict[str:Any]: the hyperparameters for the solvers.
    """
    hyperparameters = dict()

    # seed hyperparameter
    hyperparameters["seed"] = int(args.seed)

    if "Multiplicative" in args.solvers or "BRIM" in args.solvers:
        hyperparameters["capacitance"] = float(args.capacitance)
        hyperparameters["stop_criterion"] = float(args.stop_criterion)

    # Multiplicative parameters
    if "Multiplicative" in args.solvers:
        hyperparameters["num_iterations_Multiplicative"] = int(args.num_iterations_Multiplicative)
        hyperparameters["dtMult"] = float(args.dtMult)
        hyperparameters["nb_flipping"] = int(args.nb_flipping)
        hyperparameters["cluster_threshold"] = float(args.cluster_threshold)
        hyperparameters["init_cluster_size"] = float(args.init_cluster_size)
        hyperparameters["end_cluster_size"] = float(args.end_cluster_size)
        hyperparameters["cluster_choice"] = args.cluster_choice
        hyperparameters["exponent"] = float(args.exponent)
        hyperparameters["ode_choice"] = args.ode_choice
        hyperparameters["accumulation_delay"] = float(args.accumulation_delay)
        hyperparameters["broadcast_delay"] = float(args.broadcast_delay)
        hyperparameters["delay_offset"] = float(args.delay_offset)
        hyperparameters["current"] = float(args.current)
        hyperparameters["sigma_J"] = float(args.sigma)
        hyperparameters["sigma_C"] = float(args.sigma_C)

    # BRIM parameters
    if "BRIM" in args.solvers:
        hyperparameters["resistance"] = float(args.resistance)
        hyperparameters["num_iterations_BRIM"] = int(args.num_iterations_BRIM)
        hyperparameters["dtBRIM"] = float(args.dtBRIM)
        hyperparameters["do_flipping"] = bool(args.do_flipping)
        hyperparameters["coupling_annealing"] = bool(args.coupling_annealing)

    # SA-like parameters
    if "SA" in args.solvers or "SCA" in args.solvers or "DSA" in args.solvers:
        hyperparameters["initial_temp"] = float(args.T)

    if "SA" in args.solvers or "DSA" in args.solvers:
        hyperparameters["num_iterations_SA"] = int(args.num_iterations_SA)
        Tfin = float(args.T_final)
        hyperparameters["cooling_rate_SA"] = (
            return_rx(hyperparameters["num_iterations_SA"], hyperparameters["initial_temp"], Tfin)
            if hyperparameters["initial_temp"] != 0
            else 1.0
        )

    # in-Situ SA parameters
    if "inSituSA" in args.solvers:
        hyperparameters["initial_temp_inSituSA"] = float(args.initial_temp_inSituSA)
        hyperparameters["num_iterations_inSituSA"] = int(args.num_iterations_inSituSA)
        hyperparameters["nb_flips"] = float(args.nb_flips)
        Tfin = float(args.T_final_inSituSA)
        hyperparameters["cooling_rate_inSituSA"] = (
            return_rx(hyperparameters["num_iterations_inSituSA"], hyperparameters["initial_temp_inSituSA"], Tfin)
            if hyperparameters["initial_temp_inSituSA"] != 0
            else 1.0
        )

    # SCA parameters
    if "SCA" in args.solvers:
        hyperparameters["num_iterations_SCA"] = int(args.num_iterations_SCA)
        hyperparameters["q"] = float(args.q)
        hyperparameters["r_q"] = (
            return_rx(hyperparameters["num_iterations_SCA"], hyperparameters["q"], float(args.q_final))
            if hyperparameters["q"] != 0
            else 1.0
        )
        Tfin = float(args.T_final_SCA)
        hyperparameters["cooling_rate_SCA"] = (
            return_rx(hyperparameters["num_iterations_SCA"], hyperparameters["initial_temp"], Tfin)
            if hyperparameters["initial_temp"] != 0
            else 1.0
        )

    # SB parameters
    if "dSB" in args.solvers or "bSB" in args.solvers:
        hyperparameters["num_iterations_SB"] = int(args.num_iterations_SB)
        hyperparameters["dtSB"] = float(args.dtSB)
        hyperparameters["a0"] = float(args.a0)
        hyperparameters["c0"] = float(args.c0)

    # CIM parameters
    if "CIM" in args.solvers:
        hyperparameters["num_iterations_CIM"] = int(args.num_iterations_CIM)
        hyperparameters["dtCIM"] = float(args.dtCIM)
        hyperparameters["zeta"] = float(args.zeta)

    return hyperparameters


def get_best_found_gurobi(gurobi_files: list[pathlib.Path]) -> list[float]:
    """Returns a list of the best found energies in the gurobi files.

    Args:
        gurobi_files (list[pathlib.Path]): the gurobi files.

    Returns:
        list[float]: list of the best found energies.
    """
    best_found_list = []
    for file in gurobi_files:
        best_found = return_metadata(file, "solution_energy")
        best_found_list.append(best_found)
    return best_found_list


def go_over_benchmark(which_benchmark: pathlib.Path, percentage: float = 1.0, part: int = 0) -> np.ndarray:
    """Go over all the benchmarks in the given directory.

    Args:
        which_benchmark (pathlib.Path): the path to the benchmark directory.

    Returns:
        np.ndarray: a list of all the benchmarks.
    """
    optimal_energies = which_benchmark / "optimal_energy.txt"
    benchmarks = np.loadtxt(optimal_energies, dtype=str)[:, 0]
    percentage = int(len(benchmarks) * percentage)
    if (part + 1) * percentage == 1.0:
        return benchmarks[part * percentage :]
    else:
        return benchmarks[part * percentage : (part + 1) * percentage]


def return_c0(model: IsingModel) -> float:
    """Returns the optimal c0 value for simulated bifurcation.

    Args:
        model (IsingModel): the Ising model that will be solved with simulated Bifurcationl.

    Returns:
        float: the c0 hyperaparameter.
    """
    return 0.5 / (
        np.sqrt(model.num_variables)
        * np.sqrt(np.sum(np.power(model.J, 2)) / (model.num_variables * (model.num_variables - 1)))
    )


def return_G(J: np.ndarray) -> float:
    """Returns the optimal latch resistant value for the given problem.

    Args:
        J (np.ndarray): the coefficient matrix of the problem that will be solved with BRIM.

    Returns:
        float: the latch resistance.
    """
    sumJ = np.sum(np.abs(triu_to_symm(J)), axis=0)
    return np.average(sumJ) * 2


def return_q(problem: IsingModel) -> float:
    """Returns the optimal value for the penalty parameter q for the SCA solver.

    Args:
        problem (IsingModel): the problem that will be solved with SCA.

    Returns:
        float: the penalty parameter q.
    """
    eig = np.abs(spalg.eigs(triu_to_symm(-problem.J), 1)[0][0])
    return eig / 2


def compute_list_from_arg(arg: str, step: int = 1) -> np.ndarray:
    """Returns a list of integers given a argument string and step size.

    Args:
        arg (str): the argument holding the range information.
        step (int, optional): the step size. Defaults to 1.

    Returns:
        np.ndarray: the list of integers.
    """
    arg_list = arg.split()
    return np.array(range(int(arg_list[0]), int(arg_list[1]) + 1, step))
