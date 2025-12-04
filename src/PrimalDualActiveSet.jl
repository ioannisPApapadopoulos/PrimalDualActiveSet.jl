module PrimalDualActiveSet

using Gridap, SparseArrays, MatrixFactorizations
using ExtendableSparse, AlgebraicMultigrid, ILUZero
import LinearAlgebra: ldiv!, Symmetric
import IterativeSolvers: cg, gmres

export hik, ssn, bm,
    ObstacleProblem, ObstacleProblemUniform, BiactiveObstacle,
    SignoriniBox, SignoriniRectangle, ScalarSignoriniRectangle, linear_elasticity_nullsp,
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
