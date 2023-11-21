from dataclasses import dataclass


@dataclass
class Config:
    N_dim: int = 2  # Dimension of parameter space
    N_phases: int = 2

    extent: tuple = None  # Extent of parameter space to search. Set below.

    N_dist: int = 15  # Points distance function is evaluated at
    N_eval: int = 15  # Candidate points for new sample
    N_display: int = 21  # Number of points to visualise

    N_samples: int = 2000  # Samples for P_{n}
    skip_point: float = 0.8  # Min prob to entirely skip a point
    skip_phase: float = 0.00  # Min prob to skip sampling a phase
    sample_dist: float = 0.5  # Size of region to sample P_{n+1} over.
    dist_fn: str = 'abs_change'  # abs_change | KL_div

    optim_step: bool = True  # Optimise MLE when sampling x_{n+1}

    kern_var: float = 1.37
    kern_r: float = 0.37
    kern_optim: str = "ADAM" # ADAM | SGD | nothing

    def __post_init__(self):
        self.extent = ((0, 2),
                       (0, 2),
                       # (0, 1)
                       )
        self.unit_extent = tuple(((0, 1) for _ in range(self.N_dim)))


if __name__ == "__main__":
    print(Config())
