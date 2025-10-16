import numpy as np
import pathlib
import random
import time

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.flow import return_q


class SCA(SolverBase):
    def __init__(self):
        self.name = "SCA"

    def change_hyperparam(self, param: float, rate: float) -> float:
        """Changes hyperparameters according to update rule."""
        return param * rate

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp: float,
        cooling_rate_SCA: float,
        q: float,
        r_q: float,
        seed: int | None = None,
        file: pathlib.Path | None = None,
    ):
        """Implementation of the Stochastic Cellular Automata (SCA) annealing algorithm of the
        [STATICA](https://ieeexplore.ieee.org/document/9222223/?arnumber=9222223) paper

        Args:
            model (IsingModel): instance of the Ising model that needs to be optimised.
            sample (np.ndarray): initial state of the Ising model.
            num_iterations (int): total amount of iterations which the solver needs to perform.
            T (float): temperature needed for the annealing process
            r_t (float): decrease rate of the temperature
            q (float): penalty parameter to ensure the copy states are equivalent to the real states.
            r_q (float): increase rate of the penalty parameter
            seed (int, None, optional): seed to generate random numbers. Important for reproducibility.
                                        Defaults to None.
            file (pathlib.Path, None, optional): absolute path to the logger file for logging the optimisation process.
                                                 If 'None', no logging is performed.

        Returns:
            sample, energy (tuple[np.ndarray, float]): final state and energy of the optimisation process.
        """
        if q== 0.0:
            q= return_q(model)
            r_q = 1.0


        N = model.num_variables
        hs = np.zeros((N,))
        J = triu_to_symm(model.J)
        flipped_states = []
        state = np.copy(np.sign(initial_state))
        tau = np.copy(state)
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        schema = {"time": np.float32, "energy": np.float32, "state": (np.int8, (N,))}

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=state,
                    model=model,
                    num_iterations=num_iterations,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate_SCA,
                    initial_penalty=q,
                    penalty_increase=r_q,
                    seed=seed,
                )

            start_time = time.time()
            T = initial_temp
            if log.filename is not None:
                energy = model.evaluate(state.astype(np.float32))
                log.log(time=0.0, energy=energy, state=state)
            for _ in range(num_iterations):
                hs = np.matmul(J, state) + model.h # 2*N**2 + N

                Prob = self.get_prob(hs, state, q, T) # 2*N + 3*N
                rand = np.random.uniform(0, 1, size=(N,)) # N

                flipped_states = [y for y in range(N) if Prob[y] < rand[y]] # N

                tau[flipped_states] = -state[flipped_states] # N/4
                state = np.copy(tau)

                T = self.change_hyperparam(T, cooling_rate_SCA) # 1
                q = self.change_hyperparam(q, r_q) # 1
                flipped_states = []

                if log.filename is not None:
                    elapsed_time = time.time() - start_time
                    energy = model.evaluate(state.astype(np.float32))
                    log.log(time=elapsed_time, energy=energy, state=state)

            nb_operations = num_iterations * (2 * N**2 + 8 * N + N / 2 + 2)
            if log.filename is not None:
                log.write_metadata(
                    total_time=elapsed_time,
                    solution_state=state,
                    solution_energy=energy,
                    total_operations=nb_operations,
                )
            else:
                energy = model.evaluate(state.astype(np.float32))

        return state, energy, elapsed_time, nb_operations

    def get_prob(self, hs: np.ndarray, sample: np.ndarray, q: float, T: float) -> np.ndarray:
        """Calculates the probability of changing the value of the spins
           according to SCA annealing process.

        Args:
            hs (np.ndarray): local field influence.
            sample (np.ndarray): spin of the nodes.
            q (float): penalty parameter
            T (float): temperature

        Returns:
            probability (np.ndarray): probability of accepting the change of all nodes.
        """
        values = (hs * sample + q)
        probs = np.zeros_like(values)
        for i,val in enumerate(values):
            if val > 2*T:
                probs[i] = 1
            elif val < -2*T:
                probs[i] = 0
            else:
                probs[i] = val/(4*T) + 0.5
        return probs
