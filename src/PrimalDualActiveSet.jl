module PrimalDualActiveSet

using Gridap, SparseArrays, MatrixFactorizations
import LinearAlgebra: ldiv!, Symmetric

export hik, fem_hik, bm,
    ObstacleProblem, ObstacleProblemUniform,
    Signorini, SignoriniBox, SignoriniRectangle

include("nls.jl")
include("obstacleproblem.jl")
include("signorini.jl")


end # module PrimalDualActiveSet
