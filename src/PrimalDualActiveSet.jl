module PrimalDualActiveSet

using Gridap, SparseArrays, MatrixFactorizations
import LinearAlgebra: ldiv!, Symmetric

export hik, fem_hik, bm,
    ObstacleProblem, ObstacleProblemUniform,
    Signorini, SignoriniBox, SignoriniRectangle,
    OptimalControl, OptimalControlUniformSetup, 
    ControlConstrainedUniform, ControlH1ConstrainedUniform,
    StateConstrainedUniform, StateH1ConstrainedUniform

include("nls.jl")
include("obstacleproblem.jl")
include("signorini.jl")
include("optimalcontrol.jl")


end # module PrimalDualActiveSet
