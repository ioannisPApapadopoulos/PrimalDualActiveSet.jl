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
    if d ≥ 2
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