from dataclasses import dataclass


@dataclass
class Config:
    """ Experiment setup """
    N_dim: int = 3  # Dimension of parameter space
    N_phases: int = 3  # Number of phases to sample. Note: This is not checked in the code if set incorrectly.
    extent: tuple = None  # Extent of parameter space to search. Set below.

    """Search resolution"""
    N_dist: int | tuple[int, ...] = (11, 11, 11)  # Points distance function is evaluated at
    N_eval: int | tuple[int, ...] = (11, 11, 11)  # Candidate points for new sample
    N_display: int = 38  # Number of points to visualise
    reg_grid: bool = False

    """Acquisition function parameters"""
    skip_point: float = 0.95  # Min prob to be confident enough to skip a point
    skip_phase: float = 0.04  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.
    no_duplicate: bool = True # Do not duplicate points

    """GP parameters"""
    obs_prob: float = 0.95  # Default certainty of observations. (Not too close to 1 or 0)
    gaus_herm_n: int = 10  # Number of samples for Gauss Hermite quadrature
    N_optim_steps: int = 200  # Number of steps to optimise hyperparameters

    """General Parameters"""
    N_CPUs: int = 12  # Number of parallel CPU threads to use

    def __post_init__(self):
        self.extent = ((0, 1),
                       (0, 1),
                       (0, 1)
                       )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))

        # Type of grid
        if isinstance(self.N_eval, int):
            self.reg_grid = True
        elif isinstance(self.N_eval, tuple):
            self.reg_grid = False
            assert len(self.N_eval) == self.N_dim, "For irregular grids, grid dimensions must have the same number of dimensions as the parameter space"
        else:
            raise ValueError("N_eval must be either an int or a tuple of ints")

        # Assertions to make sure parameters are valid
        assert 0 < self.obs_prob < 1, "obs_P must be strictly between 0 and 1"
        assert len(self.extent) == self.N_dim, "Extent must have N_dim dimensions"


if __name__ == "__main__":
    print(Config())
