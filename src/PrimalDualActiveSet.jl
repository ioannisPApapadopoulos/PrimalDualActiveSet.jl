module PrimalDualActiveSet

using Gridap, SparseArrays, MatrixFactorizations
import LinearAlgebra: ldiv!, Symmetric

export hik, ssn, bm,
    ObstacleProblem, ObstacleProblemUniform,
    SignoriniBox, SignoriniRectangle, ScalarSignoriniRectangle,
    OptimalControlUniformSetup, 
    ControlConstrainedUniform, ControlH1ConstrainedUniform,
    StateConstrainedUniform, StateH1ConstrainedUniform,
    HyperContactRectangle

include("nls.jl")
include("problemstruct.jl")
include("obstacleproblem.jl")
include("signorini.jl")
include("hypercontact.jl")
include("optimalcontrol.jl")


end # module PrimalDualActiveSet
