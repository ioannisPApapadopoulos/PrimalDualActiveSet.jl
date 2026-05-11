# PrimalDualActiveSet.jl

This repository implements the HIK primal-dual active set strategy (https://doi.org/10.1137/S1052623401383558).

It also contains script to generate the figures and tables in

"Mesh-dependent iteration count growth in primal-dual active set strategies", Ioannis P. A. Papadopoulos, Michael Hintermüller (2026).

The primary goal of this paper was to examine iteration count growth as a consequence of the mesh-dependent convergence of HIK applied to certain discretized problems such as obstacle and Signorini problems.


## Tables & Figures

|Figure|File: examples/|
|:-:|:-:|
|1-2|[01_plot_solutions.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/01_plot_solutions.jl)|
|3a|[02_obstacle_iteration_counts.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/02_obstacle_iteration_counts.jl)|
|3b|[03_thin_obstacle_iteration_counts.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/03_thin_obstacle_iteration_counts.jl)|
|3c|[04_signorini_iteration_counts.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/04_signorini_iteration_counts.jl)|
|4|[05_obstacle_peeling.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/05_obstacle_peeling.jl)|
|5|[06_signorini_non_peeling.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/06_signorini_non_peeling.jl)|
|7|[07_second_iterate.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/07_second_iterate.jl)|

|Table|File: examples/|
|:-:|:-:|
|2, Rows 1-3|[02_obstacle_iteration_counts.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/02_obstacle_iteration_counts.jl)|
|2, Rows 4-5|[03_thin_obstacle_iteration_counts.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/03_thin_obstacle_iteration_counts.jl)|
|2, Rows 6-7|[04_signorini_iteration_counts.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/04_signorini_iteration_counts.jl)|
|3, Row 1|[08_obstacle_primal_dual_mistmatch.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/08_obstacle_primal_dual_mistmatch.jl)|
|3, Row 2|[09_obstacle_gridsequencing.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/09_obstacle_gridsequencing.jl)|
|3, Row 3-4|[10_biactive_obstacle.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/10_biactive_obstacle.jl)|
|3, Row 5-8|[11_optimal_control.jl](https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl/tree/main/examples/11_optimal_control.jl)|


## Layer-by-layer peeling

Primal-dual active set strategies applied to discretized obstacle problems feature layer-by-layer peeling. Only dofs on the boundary of the active set can switch to the inactive set at each iteration.

In the video we plot the 419 iterates of the HIK primal-dual active set applied to a 1D obstacle problem with 2^11 elements. The iterates are peeling slowly away from the obstacle.

https://github.com/user-attachments/assets/6d8ff01a-6eb8-4d92-9534-7cb5701775d6

## Installation

The package is not registered. Please install via

```pkg> add https://github.com/ioannisPApapadopoulos/PrimalDualActiveSet.jl.git```


## References

If you use this package please reference:

[1] Ioannis P. A. Papadopoulos, Michael Hintermüller. "Mesh-dependent iteration count growth in primal-dual active set strategies" (2026).


## Contact

ioannis.papadopoulos@maths.ox.ac.uk