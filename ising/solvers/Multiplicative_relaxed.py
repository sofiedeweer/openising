import numpy as np
import pathlib

# from ising.stages import LOGGER
from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class MultiplicativeRelaxed(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"

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
        self.coupling = coupling.astype(np.float32)

    def inner_loop(
        self,
        model: IsingModel,
        state: np.ndarray,
        logging: HDF5Logger | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """! Simulates the hardware with the Forward Euler method.

        @param model (IsingModel): the model to solve.
        @param state (np.ndarray): the initial state to start the simulation.
        @param logging (HDF5Logger|None): the logger to use.

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
                time=0.0,
                state=np.sign(state[: model.num_variables]),
                energy=model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32)),
                voltages=state[: model.num_variables],
            )

        # Set up new voltages
        new_state = state.copy().astype(np.float32)

        # States needed for delay calculation. The newest state is always appended to the end of the list.

        dv = self.coupling @ np.sign(state)
        dv *= self.current / self.capacitance
        state_change = True
        while i < self.num_iterations and max_change > self.stop_criterion:
            if state_change:
                dv = self.coupling @ np.sign(state)
                dv *= self.current / self.capacitance
                if self.bias:
                    dv[-1] = 0.0
            new_state = np.clip(state + self.dt * dv, -1, 1)

            if np.max(np.sign(new_state) != np.sign(state)):
                state_change = True
            else:
                state_change = False

            if i > 0 and (i % 10) == 0:
                diff = np.abs(new_state - state[-1])
                norm_prev = np.linalg.norm(state[-1])
                max_change = np.max(diff) / (norm_prev if norm_prev != 0 else 1)

            state = new_state.copy()
            i += 1
            if logging is not None:
                logging.log(
                    time=i * self.dt,
                    state=np.sign(new_state[: model.num_variables]),
                    energy=model.evaluate(np.sign(new_state[: model.num_variables]).astype(np.float32)),
                    voltages=new_state[: model.num_variables],
                )
        return (
            np.sign(new_state[: model.num_variables]),
            model.evaluate(np.sign(new_state[: model.num_variables]).astype(np.float32)),
        )

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        relaxing_iterations: int,
        current: float = 1.0,
        capacitance: float = 1.0,
        seed: int = 0,
        stop_criterion: float = 1e-8,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """!Solves the given problem using a multiplicative coupling scheme.

        @param model (IsingModel): the model to solve.
        @param initial_state (np.ndarray): the initial spins of the nodes.
        @param num_iterations (int): the number of iterations.
        @param current (float): the current flowing through a coupling unit.
        @param capacitance (float): the capacitance of the system.
        @param seed (int): the seed for random number generation.
        @param stop_criterion (float): the stopping criterion to stop the solver when the voltages don't change
        @param ode_choice (str): the choice of ODE solver.
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

        dtMult = 0.1 * capacitance / (current * np.max(np.abs(np.sum(coupling, axis=1))))

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
        if self.bias:
            v = np.empty(self.num_variables + 1, dtype=np.float32)
            v[:-1] = initial_state
            v[-1] = 1.0
        else:
            v = initial_state.astype(np.float32, copy=True)

        # Schema for logging
        if relaxing_iterations == 1:
            schema = {
                "time": np.float32,
                "energy": np.float32,
                "state": (np.int8, (self.num_variables,)),
                "voltages": (np.float32, (self.num_variables,)),
            }
        else:
            schema = {
                "energy": np.float32,
                "state": (np.int8, (self.num_variables,)),
                "cluster": (np.int32, (model.num_variables,)),
            }


        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(initial_state),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtMult,
                )
            best_energy = np.inf
            best_sample = v[: model.num_variables].copy()

            if relaxing_iterations == 1:
                logging = log
            else:
                logging = None

            for it in range(relaxing_iterations):
                sample, energy = self.inner_loop(model, v, logging)


                if self.bias:
                    v = np.block([v, np.float32(1.0)])

                # Log everything
                if log.filename is not None and relaxing_iterations > 1:
                    log.log(
                        energy=best_energy,
                        state=best_sample,
                    )

            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_time=dtMult * num_iterations,
                )
        return best_sample, best_energy, dtMult * num_iterations * relaxing_iterations, -1
