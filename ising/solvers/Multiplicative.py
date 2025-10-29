import numpy as np
import pathlib
from collections import deque

from ising.stages import LOGGER
from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class Multiplicative(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"
        self.quarter = np.float32(0.25)
        self.half = np.float32(0.5)
        self.four = np.float32(4.0)
        self.six = np.float32(6.0)

    def set_params(
        self,
        dt: float,
        num_iterations: int,
        capacitance: float,
        current: float,
        stop_criterion: float,
        coupling: np.ndarray,
    ):
        """!Set the parameters for the solver.

        @param dt (float): time step.
        @param num_iterations (int): the number of iterations.
        @param capacitance (float): the capacitance of the system.
        @param current (float): the unit current that flows through the cells.
        @param stop_criterion (float): the stopping criterion to stop the solver when the voltages stagnate.
        @param coupling (np.ndarray): the coupling matrix of the system.
        """
        self.dt = np.float32(dt)
        self.num_iterations = num_iterations
        self.capacitance = np.float32(capacitance)
        self.current = current
        self.stop_criterion = stop_criterion
        self.coupling_d = coupling.astype(np.float32)

    def construct_voltage_delay(
        self, previous_states: deque[np.ndarray], accumulation_delay: int, broadcast_delay: int
    ) -> np.ndarray:
        """!Generates a matrix of voltages taking into account what each voltage sees at the current time step.

        @param previous_states (deque[np.ndarray]): deque containing all the previous states up to\
                  accumulation_delay + broadcast_delay + 1 time steps.
        @param accumulation_delay (int): Amount of accumulation delay.
        @param broadcast_delay (int): Amount of broadcast delay.

        @return Voltages (np.ndarray): voltage matrix with all delays taken into account.
        """
        Voltages = np.zeros(
            (self.num_variables + int(self.bias), self.num_variables + int(self.bias)), dtype=np.float32
        )
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                Voltages[i, j] = previous_states[
                    int(
                        np.abs(i - j) // np.ceil((self.num_variables) / (accumulation_delay + 1))
                        + np.abs(i - j) // np.ceil((self.num_variables) / (broadcast_delay + 1))
                    )
                ][i]
        if self.bias == 1:
            Voltages[:, -1] = previous_states[0]
            Voltages[-1, :] = 1.0
        return Voltages

    def inner_loop_discrete(
        self,
        model: IsingModel,
        state: np.ndarray,
        accumulation_delay: int,
        broadcast_delay: int,
        logging: HDF5Logger | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """!Simulates a discrete version of the hardware.

        @param model (IsingModel): the model to solve.
        @param state (np.ndarray): the initial state to start the simulation with.
        @param accumulation_delay (int): Amount of accumulation delay.
        @param broadcast_delay (int): Amount of broadcast delay.
        @param logging (HDF5Logger|None): the logger to use when flipping is disabled.

        @return spins (np.ndarray): the final spins of the system.
        @return energy (float): the energy of the final state.
        @return count (np.ndarray): the count of sign changes for every spin.
        """
        if logging is not None:
            if logging.filename is not None:
                logging.log(
                    state=np.sign(state[: model.num_variables]),
                    energy=model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32)),
                    voltages=state,
                )
        LOGGER.info(
            f"Energy: {model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32))},state:{np.sign(state)}"
        )
        i = 0
        max_change = np.inf
        previous_states = deque([np.sign(state) for _ in range(accumulation_delay + broadcast_delay + 1)])
        while i < self.num_iterations and max_change > self.stop_criterion:
            Voltages = self.construct_voltage_delay(previous_states, accumulation_delay, broadcast_delay)
            dv = np.diagonal(self.coupling_d @ Voltages).copy()
            if self.bias:
                dv[-1] = 0.0

            new_state = np.sign(dv)
            if self.bias:
                new_state[-1] = 1.0

            previous_states.appendleft(np.sign(new_state))
            previous_states.pop()
            if logging is not None:
                if logging.filename is not None:
                    logging.log(
                        state=np.sign(new_state[: model.num_variables]),
                        energy=model.evaluate(np.sign(new_state[: model.num_variables]).astype(np.float32)),
                        voltages=new_state,
                    )
            if i > 0:
                max_change = np.linalg.norm(new_state - state, ord=np.inf)
            state = new_state.copy()
            i += 1
            LOGGER.info(
                f"Energy: {model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32))}, state:{state}"
            )
        return (
            state[: model.num_variables],
            model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32)),
            np.zeros((model.num_variables,), dtype=np.int32),
        )

    def inner_loop_FE(
        self,
        model: IsingModel,
        state: np.ndarray,
        accumulation_delay,
        broadcast_delay,
        logging: HDF5Logger | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """! Simulates the hardware with the Forward Euler method.

        @param model (IsingModel): the model to solve.
        @param state (np.ndarray): the initial state to start the simulation.
        @param logging (HDF5Logger|None): the logger to use when flipping is disabled.

        @return sigma (np.ndarray): the final discrete state of the system.
        @return energy (float): the final energy of the system.
        @return count (np.ndarray): the count of sign flips for every node.
        """
        # set up the simulation
        i = 0
        max_change = np.inf
        norm_prev = np.linalg.norm(state, ord=np.inf)

        if logging is not None:
            logging.log(
                state=np.sign(state[: model.num_variables]),
                energy=model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32)),
                voltages=state,
            )

        # Set up new voltages
        new_state = state.copy().astype(np.float32)
        count = np.zeros((model.num_variables,), dtype=np.int32)

        # States needed for delay calculation. The newest state is always appended to the end of the list.
        previous_states = deque([np.sign(state) for _ in range(accumulation_delay + 1 + broadcast_delay)])

        while i < self.num_iterations and max_change > self.stop_criterion:
            Voltages = self.construct_voltage_delay(previous_states, accumulation_delay, broadcast_delay)
            dv = np.diagonal(self.coupling_d @ Voltages).copy()*self.current / self.capacitance
            if self.bias:
                dv[-1] = 0.0
            new_state = state + self.dt * dv
            new_state = np.clip(new_state, -1, 1)

            previous_states.appendleft(np.sign(new_state))
            previous_states.pop()
            count = np.where(np.sign(new_state) != np.sign(state), count + 1, count)

            if i > 0 and i % 100:
                diff = np.abs(new_state - previous_states[-1])
                max_change = np.max(diff) / (norm_prev if norm_prev != 0 else 1)
                norm_prev = np.linalg.norm(new_state, ord=np.inf)

            state = new_state.copy()
            i += 1
            if logging is not None:
                logging.log(
                    state=np.sign(new_state[: model.num_variables]),
                    energy=model.evaluate(np.sign(new_state[: model.num_variables]).astype(np.float32)),
                    voltages=new_state,
                )
        return (
            np.sign(new_state[: model.num_variables]),
            model.evaluate(np.sign(new_state[: model.num_variables]).astype(np.float32)),
            count,
        )

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        dtMult: float,
        num_iterations: int,
        nb_flipping: int,
        cluster_threshold: float,
        init_cluster_size: float,
        end_cluster_size: float,
        exponent: float = 3.0,
        cluster_choice: str = "random",
        current: float = 1.0,
        capacitance: float = 1.0,
        seed: int = 0,
        stop_criterion: float = 1e-8,
        ode_choice: str = "RK",
        accumulation_delay: int = 1,
        broadcast_delay: int = 0,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """!Solves the given problem using a multiplicative coupling scheme.

        @param model (IsingModel): the model to solve.
        @param initial_state (np.ndarray): the initial spins of the nodes.
        @param dtMult (float): time step.
        @param num_iterations (int): the number of iterations.
        @param nb_flipping (int): the number of flipping iterations.
        @param cluster_threshold (float): the threshold for clustering.
        @param init_cluster_size (float): the initial cluster size.
        @param end_cluster_size (float): the final cluster size.
        @param exponent (float): the exponent for the exponential decrease of the cluster size.
        @param cluster_choice (str): the choice of clustering method.
        @param pseudo_length (int | None): the sequence length of the pseudo-random number generator.
        @param resistance (float): the resistance of the system.
        @param capacitance (float): the capacitance of the system.
        @param seed (int): the seed for random number generation.
        @param stop_criterion (float): the stopping criterion to stop the solver when the voltages don't change
        @param ode_choice (str): the choice of ODE solver.
        @param delay (int): the amount of delay of the accumulation and broadcasting delay.
        @param file: (pathlib.Path,None): the path to the logfile

        @return state (np.ndarray): the final state of the system.
        @return energy (float): best energy of the system.
        @return computation_time (float): total computation time for analog simulation.
        @return operation_count (int): the number of operations performed.
        """

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            self.bias = np.int8(1)
        else:
            new_model = model
            self.bias = np.int8(0)

        # Ensure the mean and variance of J are reasonable
        coupling = triu_to_symm(new_model.J)
        self.num_variables = model.num_variables

        sum_J = np.sum(coupling, axis=1)
        dtMult = 1 / np.max(sum_J)
        LOGGER.debug(f"Adjusted time step to {dtMult:.4e} for stability.")

        # Set the parameters for easy calling
        init_size = int(init_cluster_size * model.num_variables)
        end_size = int(end_cluster_size * model.num_variables)
        self.set_params(
            dtMult,
            num_iterations,
            capacitance,
            current,
            stop_criterion,
            coupling,
        )

        # make sure the correct random seed is used
        np.random.seed(seed)
        self.generator = np.random.choice

        # Set up the bias node and add noise to the initial voltages
        num_var = model.num_variables
        if self.bias:
            v = np.empty(num_var + 1, dtype=np.float32)
            v[:-1] = initial_state
            v[-1] = 1.0
        else:
            v = initial_state.astype(np.float32, copy=True)

        # Schema for logging
        if nb_flipping == 1:
            schema = {
                "energy": np.float32,
                "state": (np.int8, (num_var,)),
                "voltages": (np.float32, (num_var + int(self.bias),)),
            }
        else:
            schema = {
                "energy": np.float32,
                "state": (np.int8, (num_var,)),
            }

        # Define cluster function
        choice = ""
        if cluster_choice == "random":
            find_cluster = self.find_cluster_random
        elif cluster_choice == "gradient":
            find_cluster = self.find_cluster_gradient
        elif cluster_choice == "weighted_mean_smallest":
            find_cluster = self.find_cluster_weighted_mean
            choice = "smallest"
        elif cluster_choice == "weighted_mean_largest":
            find_cluster = self.find_cluster_weighted_mean
            choice = "largest"
        elif cluster_choice == "frequency":
            find_cluster = self.find_cluster_frequency
        else:
            raise ValueError(
                f" Unknown cluster choice: {cluster_choice}. \
             Currently supported: random, gradient, weighted_mean_smallest, weighted_mean_largest, frequency."
            )
        additional_information = {
            "count": np.ndarray,
            "current_state": np.ndarray,
            "cluster_threshold": cluster_threshold,
            "optimal_points": [],
            "choice": choice,
        }

        if ode_choice == "FE":
            inner_loop = self.inner_loop_FE
        elif ode_choice == "discrete":
            inner_loop = self.inner_loop_discrete
        else:
            raise ValueError(f"Unknown ODE choice: {ode_choice}.")

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(initial_state),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtMult,
                    cluster_choice=cluster_choice,
                    exponent=exponent,
                )
            best_energy = np.inf
            best_sample = v[: model.num_variables].copy()
            if nb_flipping == 1:
                logging = log
            else:
                logging = None

            counter = 0  # Counter for no improvement
            restart = 0  # When counter reaches threshold, size of cluster is reset with restart = it
            for it in range(nb_flipping):
                sample, energy, count = inner_loop(model, v, accumulation_delay, broadcast_delay, logging)
                additional_information["count"] = count
                additional_information["current_state"] = sample

                if energy < best_energy:
                    best_energy = energy
                    best_sample = sample.copy()
                    additional_information["optimal_points"].append((best_sample.copy(), best_energy))
                    counter = 0
                else:
                    counter += 1

                # if counter >= int(nb_flipping / 4):
                #     restart = int(it / 2)
                #     counter = 0

                cluster = find_cluster(
                    self.size_function(
                        iteration=it - restart,
                        total_iterations=nb_flipping + int(nb_flipping == 1),
                        init_size=init_size,
                        end_size=end_size,
                        exponent=exponent,
                    ),
                    **additional_information,
                )
                v = best_sample.copy()
                v[cluster] *= -1
                if self.bias:
                    v = np.block([v, np.float32(1.0)])

                # Log everything
                if log.filename is not None and nb_flipping > 1:
                    if nb_flipping > 1:
                        log.log(energy=best_energy, state=best_sample)
                    else:
                        log.log(energy=best_energy, state=best_sample, voltages=best_sample)

            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_time=dtMult * num_iterations,
                )
        return best_sample, best_energy, dtMult * num_iterations * nb_flipping, -1

    def size_function(
        self,
        iteration: int,
        total_iterations: int,
        init_size: int,
        end_size: int,
        exponent: float = 3.0,
    ):
        return int(
            np.floor(
                (((end_size - 1) / init_size) ** (iteration * exponent / (total_iterations - 1)))
                * (init_size - end_size)
                + end_size
            )
        )

    def find_cluster_gradient(self, cluster_size: int, **additional_information) -> np.ndarray:
        coupling = self.coupling_d * self.resistance
        sigma = additional_information["current_state"]
        threshold = additional_information["cluster_threshold"]

        gradient = (coupling @ np.block([sigma, 1]))[: len(sigma)]
        gradient /= np.max(gradient)
        available_nodes = np.where(gradient >= threshold, np.arange(len(sigma)), -1)  # Chosen nodes based on threshold
        if len(available_nodes[available_nodes >= 0]) < cluster_size:  # Case when not enough nodes are available
            current_size = len(available_nodes[available_nodes >= 0])
            ind_unavailable_nodes = np.where(available_nodes < 0)[0]
            chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
            available_nodes[chosen_nodes] = np.arange(len(sigma))[chosen_nodes]
            cluster = available_nodes[available_nodes >= 0]
        else:  # case when enough nodes are available
            cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(cluster_size,), replace=False)
        return cluster

    def find_cluster_random(self, cluster_size: int, **additional_information) -> np.ndarray:
        """Finds a random cluster of nodes to flip.

        Args:
            cluster_size (int): the size of the cluster to find.
        """
        cluster = self.generator(np.arange(self.num_variables), size=(cluster_size,), replace=False)
        return cluster

    def find_cluster_weighted_mean(self, cluster_size: int, **additional_information) -> np.ndarray:
        optimal_points = additional_information["optimal_points"]
        choice = additional_information["choice"]
        weight_nodes = np.zeros_like(optimal_points[0][0], dtype=float)
        for point, en in optimal_points:
            weight_nodes += 1 / en * point  # the smaller the energy, the larger the weight
        if np.linalg.norm(weight_nodes) == 0:
            weight_nodes = np.random.random(weight_nodes.shape)  # First step is random choice
        weight_nodes = np.abs(weight_nodes) / np.max(np.abs(weight_nodes))
        if choice == "smallest":
            available_nodes = np.where(weight_nodes < additional_information["cluster_threshold"])[0]
            current_size = len(available_nodes)
            if len(available_nodes) < cluster_size:
                ind_unavailable_nodes = np.where(weight_nodes >= additional_information["cluster_threshold"])[0]
                chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
                available_nodes = np.append(available_nodes, chosen_nodes)
        else:
            available_nodes = np.where(weight_nodes > additional_information["cluster_threshold"])[0]
            current_size = len(available_nodes)
            if len(available_nodes) < cluster_size:
                ind_unavailable_nodes = np.where(weight_nodes <= additional_information["cluster_threshold"])[0]
                chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
                available_nodes = np.append(available_nodes, chosen_nodes)
        cluster = np.random.choice(available_nodes, size=(cluster_size,), replace=False)
        return cluster

    def find_cluster_frequency(self, cluster_size: int, **additional_information) -> np.ndarray:
        """Finds the cluster of nodes to flip. These nodes are chosen based on the frequency of flipping.

        Args:
            counts (np.ndarray): the amount of times each node changed their sign during convergence.
            cluster_size (int): the size of the cluster to find.
            cluster_threshold (float): the threshold for selecting nodes.
        """
        counts = additional_information["counts"]
        cluster_threshold = additional_information["cluster_threshold"]

        freq = counts / (np.max(np.abs(counts)) if np.max(np.abs(counts)) != 0 else 1)

        available_nodes = np.where(freq < cluster_threshold)[0]
        current_size = len(available_nodes)
        if len(available_nodes) < cluster_size:
            ind_unavailable_nodes = np.where(freq >= cluster_threshold)[0]
            chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
            available_nodes = np.append(available_nodes, chosen_nodes)
            cluster = available_nodes
        else:
            cluster = np.random.choice(available_nodes, size=(cluster_size,), replace=False)
        return cluster
