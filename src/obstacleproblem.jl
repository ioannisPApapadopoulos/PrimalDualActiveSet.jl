struct ObstacleProblem{T}
    model::DiscreteModel
    labels::Gridap.Geometry.FaceLabeling
    V::Gridap.FESpaces.UnconstrainedFESpace
    U::TrialFESpace
    Ω::Gridap.Geometry.BodyFittedTriangulation
    dΩ::Gridap.CellData.GenericMeasure
    res_jac::Tuple{Function, Function}
    op::Gridap.FESpaces.FEOperatorFromWeakForm
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

function ObstacleProblemUniform(n::Int, f::Function, φ::Function; d::Int=1)
    if d == 1
        domain = (0,1)
        partition = (n,)
    elseif d == 2
        domain = (0,1,0,1)
        partition = (n,n)
    elseif d == 3
        domain = (0,1,0,1,0,1)
        partition = (n,n,n)
    else
        error("Not implemented for d=$d.")
    end
    model = CartesianDiscreteModel(domain,partition)
    if d==2
        model = simplexify(model)
    end
    reffe_u = ReferenceFE(lagrangian,Float64,1)
    labels = get_face_labeling(model)
    V = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
    U = TrialFESpace(V, 0.0)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)
    a(u, v) =∫(∇(u) ⋅ ∇(v) - f ⋅ v) * dΩ
    j(u, du, v) =∫(∇(du) ⋅ ∇(v)) * dΩ
    op = FEOperator(a, j, U, V)

    lb = -1e10*ones(V.nfree)
    ub = interpolate_everywhere(φ, V).free_values
    return ObstacleProblem{Float64}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end

function hik(P::ObstacleProblem, u0::Gridap.FESpaces.SingleFieldFEFunction; max_iter::Integer=1000, history::Bool=false)
    hik(P.op, u0, P.lb, P.ub, max_iter=max_iter, sym_pos_def=true, history=history)
end

function ssn(P::ObstacleProblem, M::AbstractMatrix{T}, u0::Gridap.FESpaces.SingleFieldFEFunction; max_iter::Integer=1000, history::Bool=false) where T
    ssn(P.op, u0, P.lb, P.ub, M, max_iter=max_iter, history=history)
end