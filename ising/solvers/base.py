from abc import ABC, abstractmethod
import numpy as np

from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


class SolverBase(ABC):
    """Abstract Base Class for Ising solvers."""

    def __init__(self):
        self.name= ""
        self.zero_en_length = 50
        self.max_energy_change = 1e-6


    @abstractmethod
    def solve(self, model: IsingModel):
        pass

    def handle_stop_criterion(self, old, new, order=None):
        return np.linalg.norm(old-new, ord=order)/np.linalg.norm(old, ord=order)

    def log_metadata(
        self,
        logger: HDF5Logger,
        initial_state: np.ndarray,
        model: IsingModel,
        num_iterations: int,
        **kwargs,
    ):
        """Log metadata to the HDF5 file. This metadata is mostly needed for postprocessing and
            thus common to most solvers.

        Args:
            logger (HDF5Logger): the logger object to use for logging metadata
            initial_state (np.ndarray): the initial state of the system
            model (IsingModel): the model to be optimized
            num_iterations (int): the number of iterations to run the solver for
        """
        metadata = {
            "solver": self.name,
            "problem_size": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
        logger.write_metadata(**metadata)
