struct ObstacleProblem{T}
    model::DiscreteModel
    labels::Gridap.Geometry.FaceLabeling
    V::Union{Gridap.FESpaces.UnconstrainedFESpace, MultiFieldFESpace}
    U::Union{TrialFESpace,MultiFieldFESpace}
    Ω::Gridap.Geometry.BodyFittedTriangulation
    dΩ::Gridap.CellData.GenericMeasure
    res_jac::Tuple{Function, Function}
    op::Gridap.FESpaces.FEOperatorFromWeakForm
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

struct NonSymmetricObstacleProblem{T}
    model::DiscreteModel
    labels::Gridap.Geometry.FaceLabeling
    V::Union{Gridap.FESpaces.UnconstrainedFESpace, MultiFieldFESpace}
    U::Union{TrialFESpace,MultiFieldFESpace}
    Ω::Gridap.Geometry.BodyFittedTriangulation
    dΩ::Gridap.CellData.GenericMeasure
    res_jac::Tuple{Function, Function}
    op::Gridap.FESpaces.FEOperatorFromWeakForm
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

function hik(P::ObstacleProblem, u0::Gridap.FESpaces.SingleFieldFEFunction; max_iter::Integer=1000, krylov_solver::Bool=false, history::Bool=false, nullsp=Vector{Float64}())
    hik(P.op, u0, P.lb, P.ub, max_iter=max_iter, sym_pos_def=true, krylov_solver=krylov_solver, history=history, nullsp=nullsp)
end

function ssn(P::ObstacleProblem, M::AbstractMatrix{T}, u0::Gridap.FESpaces.SingleFieldFEFunction; max_iter::Integer=1000, history::Bool=false) where T
    ssn(P.op, u0, P.lb, P.ub, M, max_iter=max_iter, history=history)
end

function hik(P::NonSymmetricObstacleProblem, z0::Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction}; max_iter::Integer=1000, history::Bool=false)
    hik(P.op, z0, P.lb, P.ub, max_iter=max_iter, history=history)
end