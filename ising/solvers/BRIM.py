import time
import numpy as np
import pathlib

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm

###
# This code is adapted from the code of the BRIM paper: https://ieeexplore.ieee.org/document/9407038/,
# given to us by professor Michael Huang.
###


class BRIM(SolverBase):
    def __init__(self):
        self.name = "BRIM"
        self.p = 0.02
        self.end_prob = 1e-6
        self.fourth = np.float32(1 / 4)
        self.three = np.float32(3)
        self.two_thirds = np.float32(2 / 3)

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
        if self.bias:
            self.chosen_flips[-1] = False  # Do not flip the bias node

        self.flip_voltages = np.where(self.chosen_flips, -np.abs(voltages), 0).astype(np.float32)

        self.p += self.prob_change

    def dvdt(self, t, vt, coupling, Ka):
        # Make sure the bias node is 1
        if self.bias:
            vt[-1] = 1.0

        # ZIV diode
        z = vt / self.resistance + (
            (-2.156334025305975e-05 * np.power(vt, 5))
            + (1.017179575405042e-04 * np.power(vt, 3))
            + (-2.231312342175098e-05 * vt)
        )

        # Compute the differential equation
        if self.do_flipping:
            flip = (self.flip_voltages - vt) / self.resistance
            dv = np.where(self.chosen_flips, flip / self.capacitance, 1 / self.capacitance * ((Ka * coupling) @ vt - z))
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
        """Simulates the BRIM dynamics by integrating the Lyapunov equation through time with the RK4 method.

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
            self.bias = True
        else:
            new_model = model
            self.bias = False
        J = triu_to_symm(new_model.J).astype(np.float32) / resistance

        # Make sure the correct seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Ensure the bias node is added and add noise to the initial voltages
        if self.bias:
            v = np.block([initial_state, 1.0])
        else:
            v = initial_state
        v = v.astype(np.float32)
        # Schema for the logging
        schema = {
            "time_clock": float,
            "energy": np.float32,
            "voltages": (np.float32, (model.num_variables,)),
        }

        # Set up the flipping probability
        self.prob_change = (self.p - self.end_prob) / (num_iterations - 1)

        # Set all parameters
        self.num_variables = model.num_variables
        self.capacitance = np.float32(capacitance)
        self.resistance = np.float32(resistance)
        self.do_flipping = do_flipping
        self.chosen_flips = np.zeros(self.num_variables, dtype=bool)
        self.flip_voltages = np.zeros(self.num_variables, dtype=np.float32)

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
                log.log(time_clock=0.0, energy=energy, voltages=v[: model.num_variables])

            while i < (num_iterations) and max_change > stop_criterion:
                tk = t_eval[i]

                if coupling_annealing:
                    Ka = self.Ka(tk, tend)
                else:
                    Ka = np.float32(1.0)

                # Forward Euler
                k1 = dtBRIM * self.dvdt(tk, previous_voltages, J, Ka)

                new_voltages = np.clip(
                    previous_voltages + k1, np.float32(-1), np.float32(1)
                )

                if self.do_flipping:
                    self.choose_spinflips(new_voltages)
                # Log everything
                if log.filename is not None:
                    sample = np.sign(new_voltages[: model.num_variables])
                    energy = model.evaluate(sample)
                    log.log(time_clock=tk, energy=energy, voltages=new_voltages[: model.num_variables])

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
                    log.log(time_clock=tk, energy=energy, voltages=new_voltages[: model.num_variables])
            if log.filename is not None:
                log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
            else:
                energy = model.evaluate(np.sign(new_voltages[: model.num_variables]))
        return sample, energy, tend, -1
