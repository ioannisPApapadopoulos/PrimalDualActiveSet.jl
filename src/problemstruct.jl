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

struct NonlinearNonSymmetricObstacleProblem{T}
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

function hik(P::ObstacleProblem, u0::Gridap.FESpaces.SingleFieldFEFunction; solver_flag::Val=Val(1),tol::Float64=1e-9, max_iter::Integer=1000, history::Bool=false, nullsp=Vector{Float64}())
    hik(P.op, u0, P.lb, P.ub, Val(true), Val(true), solver_flag=solver_flag, max_iter=max_iter, tol=tol, history=history, nullsp=nullsp)
end

function ssn(P::ObstacleProblem, M::AbstractMatrix{T}, u0::Gridap.FESpaces.SingleFieldFEFunction; max_iter::Integer=1000, history::Bool=false) where T
    ssn(P.op, u0, P.lb, P.ub, M, Val(true), max_iter=max_iter, history=history)
end

function hik(P::NonSymmetricObstacleProblem, z0::Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction}; solver_flag::Val=Val(1), max_iter::Integer=1000, history::Bool=false)
    hik(P.op, z0, P.lb, P.ub, Val(true), Val(false), solver_flag=solver_flag, max_iter=max_iter, history=history)
end

function hik(P::NonlinearNonSymmetricObstacleProblem, z0::Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction}; solver_flag::Val=Val(1), max_iter::Integer=1000, history::Bool=false)
    hik(P.op, z0, P.lb, P.ub, Val(false), Val(false), solver_flag=solver_flag, max_iter=max_iter, history=history)
end