import time
import numpy as np
import pathlib

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class BRIM(SolverBase):
    def __init__(self):
        self.name = "BRIM"

    def Ka(self, time: float, end_time: float) -> float:
        """Returns the coupling annealing term.

        Args:
            time (float): the time.
            end_time (float): the end time.
        Returns:
            Ka (float): the coupling annealing term.
        """
        return np.float32(1 - np.exp(-time / end_time))

    def choose_spinflips(self, voltages: np.ndarray):
        random_v = np.random.uniform(0, 1, (self.num_variables))
        self.chosen_flips = random_v < self.p

        self.flip_voltages = np.where(self.chosen_flips, -np.abs(voltages), 0).astype(np.float32)

        self.p += self.prob_change

    def dvdt(self, t, vt, coupling, Ka):
        # Make sure the bias node is 1
        if self.bias:
            vt[-1] = 1.0

        # ZIV diode
        z = vt * (vt + 1) * (vt - 1) / self.resistance

        # Compute the differential equation
        if self.do_flipping:
            flip = (self.flip_voltages - vt) / self.resistance
            dv = np.where(
                self.chosen_flips, flip / self.capacitance, 1 / self.capacitance * ((Ka * coupling) @ vt - z)
            )
        else:
            dv = 1 / self.capacitance * ((Ka * coupling) @ vt - z)

        # Make sure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt > 1)
        cond2 = (dv < 0) & (vt < -1)
        dv *= np.where(cond1 | cond2, 0.0, 1)

        # Make sure the bias node does not change
        if self.bias:
            dv[-1] = 0.0
        return dv

    def set_parameters(
        self,
        model: IsingModel,
        capacitance: float,
        resistance: float,
        initial_prob: float,
        prob_change: float,
        bias: bool,
        do_flipping: bool,
    ):
        self.num_variables = model.num_variables
        self.capacitance = np.float32(capacitance)
        self.resistance = np.float32(resistance)
        self.p = initial_prob
        self.prob_change = prob_change
        self.bias = bias
        self.do_flipping = do_flipping
        self.chosen_flips = np.zeros(self.num_variables, dtype=bool)
        self.flip_voltages = np.zeros(self.num_variables, dtype=np.float32)
        self.fourth = np.float32(1 / 4)
        self.three = np.float32(3)
        self.two_thirds = np.float32(2 / 3)

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtBRIM: float,
        capacitance: float,
        resistance: float,
        stop_criterion: float = 1e-8,
        file: pathlib.Path | None = None,
        coupling_annealing: bool = False,
        do_flipping: bool = False,
        seed: int = 0,
    ) -> tuple[np.ndarray, float]:
        """Simulates the BLIM dynamics by integrating the Lyapunov equation through time with the RK4 method.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            initial_state (np.ndarray): initial spins of the nodes
            num_iterations (int): amount of iterations that need to be simulated
            dtBRIM (float): time step.
            capacitance (float): capacitor parameter.
            stop_criterion (float, optional): stop criterion for the maximum allowed change between iterations.
                                              Defaults to 1e-8.
            file (pathlib.Path, None, Optional): absolute path to which data will be logged. If 'None',
                                                 nothing is logged.
            coupling_annealing (bool, optional): whether to anneal the coupling matrix. Defaults to False.
            seed (int, optional): seed for the random number generator. Defaults to 0.

        Returns:
            sample,energy (tuple[np.ndarray, float]): the final state and energy of the system.
        """

        # Set the time evaluations
        dtBRIM = np.float32(dtBRIM)
        tend = dtBRIM * num_iterations
        t_eval = np.linspace(0.0, tend, num_iterations, dtype=np.float32)

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            bias = True
        else:
            new_model = model
            bias = False
        J = triu_to_symm(new_model.J).astype(np.float32) / resistance

        # Make sure the correct seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Ensure the bias node is added and add noise to the initial voltages
        if bias:
            v = np.block([initial_state, 1.0])
        else:
            v = initial_state
        v = v.astype(np.float32)
        # Schema for the logging
        schema = {
            "time_clock": float,
            "energy": np.float32,
            "state": (np.int8, (model.num_variables,)),
            "voltages": (np.float32, (model.num_variables,)),
        }

        # Set up the flipping probability
        init_prob = 0.02
        end_prob = 1e-6
        prob_change = (init_prob - end_prob) / (num_iterations - 1)

        # Store all parameters
        self.set_parameters(model, capacitance, resistance, init_prob, prob_change, bias, do_flipping)

        with HDF5Logger(file, schema) as log:
            # Log the initial metadata
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(v),
                    model=model,
                    num_iterations=num_iterations,
                    C=capacitance,
                    time_step=dtBRIM,
                    seed=seed,
                    stop_criterion=stop_criterion,
                    coupling_annealing=coupling_annealing,
                )

            # Initialize the simulation variables
            i = 0
            previous_voltages = np.copy(v).astype(np.float32)
            max_change = np.inf

            # Initial logging
            if log.filename is not None:
                sample = np.sign(v[: model.num_variables])
                energy = model.evaluate(sample)
                log.log(time_clock=0.0, energy=energy, state=sample, voltages=v[: model.num_variables])

            while i < (num_iterations) and max_change > stop_criterion:
                tk = t_eval[i]

                if coupling_annealing:
                    Ka = self.Ka(tk, tend)
                else:
                    Ka = np.float32(1.0)

                # Runge Kutta steps, k1 is the derivative at time step t, k2 is the derivative at time step t+2/3*dt
                k1 = dtBRIM * self.dvdt(tk, previous_voltages, J, Ka)
                k2 = dtBRIM * self.dvdt(tk + self.two_thirds * dtBRIM, previous_voltages + self.two_thirds * k1, J, Ka)

                new_voltages = np.clip(
                    previous_voltages + self.fourth * (k1 + self.three * k2), np.float32(-1), np.float32(1)
                )

                if self.do_flipping:
                    self.choose_spinflips(new_voltages)
                # Log everything
                if log.filename is not None:
                    sample = np.sign(new_voltages[: model.num_variables])
                    energy = model.evaluate(sample)
                    log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[: model.num_variables])

                # Update criterion changes
                if i > 0:
                    max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                        previous_voltages, ord=np.inf
                    )
                previous_voltages = np.copy(new_voltages)
                i += 1

            # Make sure to log to the last iteration if the stop criterion is reached
            if max_change < stop_criterion and log.filename is not None:
                for j in range(i, num_iterations):
                    tk = t_eval[j]
                    log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[: model.num_variables])
            if log.filename is not None:
                log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
            else:
                energy = model.evaluate(np.sign(new_voltages[: model.num_variables]))
        return sample, energy, tend, -1
