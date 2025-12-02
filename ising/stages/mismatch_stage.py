from ising.stages import LOGGER
from typing import Any
import numpy as np
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel


class MismatchStage(Stage):
    """! Stage to inject the noise on the ising model."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        config: Any,
        ising_model: IsingModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.ising_model = ising_model

    def run(self):
        if self.config.mismatch_std > 0.0:
            mismatch_J = self.ising_model.J * (
                1 + np.random.normal(0, self.config.mismatch_std, self.ising_model.J.shape)
            )

            mismatch_h = self.ising_model.h * (
                1 + np.random.normal(0, self.config.mismatch_std, self.ising_model.h.shape)
            )

            mismatched_model = IsingModel(mismatch_J, mismatch_h, self.ising_model.c)
        else:
            LOGGER.debug("Mismatch is disabled, using original J/h matrices.")
            mismatched_model = self.ising_model

        self.kwargs["ising_model"] = mismatched_model
        self.kwargs["config"] = self.config
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug in sub_stage.run():
            ans.ising_model = self.ising_model
            ans.mismatched_model = mismatched_model
            for solver in ans.config.solvers:
                for energy_id in range(len(ans.energies[solver])):
                    ans.energies[solver][energy_id] = self.ising_model.evaluate(
                        ans.states[solver][energy_id].astype(np.float32)
                    )
                    if hasattr(ans, "tsp_energies"):
                        if ans.tsp_energies[energy_id] == np.inf:
                            ans.tsp_energies[energy_id] = np.inf
                        else:
                            ans.tsp_energies[energy_id] = ans.energies[energy_id]
            yield ans, debug
