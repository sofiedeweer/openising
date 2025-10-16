import numpy as np
import pathlib
import time
from abc import abstractmethod

from ising.stages.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.flow import return_c0


class SB(SolverBase):
    """Implements discrete Simulated bifurcation as is seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
    This implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm.

    This class inherits from the abstract Solver base class.
    """

    def __init__(self):
        self.name = "SB"

    def update_x(self, y, dt, a0):
        return a0 * y * dt

    def update_rule(self, x, y, node):
        x[node] = np.sign(x[node])
        y[node] = 0.0

    def at(self, t, a0, dt, num_iterations) -> float:
        return a0 / (dt * num_iterations) * t

    @abstractmethod
    def solve(self, model: IsingModel):
        pass


class ballisticSB(SB):
    def __init__(self):
        super().__init__()
        self.name = f"b{self.name}"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        c0: float,
        dtSB: float,
        a0: float = 1.0,
        seed: int = 0,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """Performs the ballistic Simulated Bifurcation algorithm first proposed by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
        This variation of Simulated Bifurcation introduces perfectly inelastic walls at |x_i| = 1
        to reduce analog errors.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            x (np.ndarray): the initial position of the nonlinear oscillators.
            y (np.ndarray): the initial momenta of the nonlinear oscillators.
            num_iterations (int): amount of iterations that needs to be performed.
            a0 (float, Optional): hyperparameter. Defaults to 1.
            at (callable): changing hyperparameter that induces the bifurcation.
            c0 (float): hyperparameter.
            dt (float): time step.
            file (pathlib.Path, None, Optional): full path to which data will be logged. If 'None',
                                                 no logging is performed
            bit_width (int, optional): The bit width for the position and momenta. Defaults to 16.

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        N = model.num_variables

        if c0 == 0.0:
            c0 = return_c0(model)
        if seed == 0:
            seed = time.time()
        np.random.seed(seed)

        # Set up the model and initial states with the correct data type
        J = np.array(triu_to_symm(model.J), dtype=np.float32)
        h = np.array(model.h)
        initial_state = np.array(initial_state)
        x = np.zeros_like(initial_state, dtype=np.float32)
        y = np.random.uniform(-0.1, 0.1, (model.num_variables,)).astype(np.float32)

        schema = {
            "time": np.float32,
            "energy": float,
            "positions": (np.float32, (N,)),
        }

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(x),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtSB,
                    a0=a0,
                    c0=c0,
                )

            tk = 0.0
            if log.filename is not None:
                sample = np.sign(x)
                energy = model.evaluate(sample)
                log.log(time=0.0, energy=energy, positions=x)
            start_time = time.time()
            for _ in range(num_iterations):
                atk = self.at(tk, a0, dtSB, num_iterations) # 4

                y += (-(a0 - atk) * x + c0 * np.matmul(J, x) + c0 * h) * dtSB
                # 1 + N + 2*N**2 + N + N + 2*N + N= 2*N**2 + 6*N + 1
                x += self.update_x(y, dtSB, a0) # N+1

                y = np.where(np.abs(x) >= 1, 0, y) # N
                x = np.where(np.abs(x) >= 1, np.sign(x), x) # N

                tk += dtSB # 1
                if log.filename is not None:
                    sample = np.sign(x)
                    energy = model.evaluate(sample)
                    elapsed_time = time.time() - start_time
                    log.log(time=elapsed_time, energy=energy, positions=x)

            nb_operations = num_iterations * (2 * N**2 + 9 * N + 6)
            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_operations=nb_operations,
                    total_time=elapsed_time,
                )
            else:
                sample = np.sign(x)
                energy = model.evaluate(sample)
        return sample, energy, elapsed_time, nb_operations


class discreteSB(SB):
    def __init__(self):
        super().__init__()
        self.name = f"d{self.name}"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        c0: float,
        dtSB: float,
        a0: float = 1.0,
        seed: int = 0,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """Performs the discrete Simulated Bifurcation algorithm first proposed by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
        This variation of Simulated Bifurcation discretizes the positions x_i at all times to reduce analog errors.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            x (np.ndarray): the initial position of the nonlinear oscillators.
            y (np.ndarray): the initial momenta of the nonlinear oscillators.
            num_iterations (int): amount of iterations that needs to be performed.
            a0 (float, Optional): hyperparameter. Defaults to 1.
            at (callable): changing hyperparameter that induces the bifurcation.
            c0 (float): hyperparameter.
            dt (float): time step.
            file (pathlib.Path, None, Optional): full path to which data will be logged. If 'None',
                                                 no logging is performed

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        N = model.num_variables
        tk = 0.0
        if c0 == 0.0:
            c0 = return_c0(model)

        if seed == 0:
            seed = time.time()
        np.random.seed(seed)

        # Set up the model and initial states with the correct data type
        J = np.array(triu_to_symm(model.J), dtype=np.float32)
        h = np.array(model.h)
        initial_state = np.array(initial_state)
        x = np.zeros_like(initial_state, dtype=np.float32)
        y = np.random.uniform(-0.1, 0.1, (model.num_variables,)).astype(np.float32)

        schema = {
            "time": np.float32,
            "energy": np.float32,
            "positions": (np.float32, (N,)),
        }

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(x),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtSB,
                    a0=a0,
                    c0=c0,
                )
                sample = np.sign(x)
                energy = model.evaluate(sample)
                log.log(time=0.0, energy=energy, positions=x)
            start_time = time.time()
            for i in range(num_iterations):
                atk = self.at(tk, a0, dtSB, num_iterations) # 3

                y += (-(a0 - atk) * x + c0 * np.matmul(J, np.sign(x)) + c0 * h) * dtSB
                # 1+N + 2*N**2 + N + N + 2*N + N = 2*N**2 + 6*N + 1
                x += self.update_x(y, dtSB, a0) # N+1

                for j in range(N): # N
                    if np.abs(x[j]) > 1:
                        self.update_rule(x, y, j)

                tk += dtSB # 1
                if log.filename is not None:
                    elapsed_time = time.time() - start_time
                    sample = np.sign(x)
                    energy = model.evaluate(sample)
                    log.log(time=elapsed_time, energy=energy, positions=x)

            nb_operations = num_iterations * (2 * N**2 + 9 * N + 5)
            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_operations=nb_operations,
                    total_time=elapsed_time,
                )
            else:
                sample = np.sign(x)
                energy = model.evaluate(np.sign(x))
        return sample, energy, elapsed_time, nb_operations
