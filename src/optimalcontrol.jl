function OptimalControlUniformSetup(n::Int, ud::Function, β::T) where T

    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)

    reffe_u = ReferenceFE(lagrangian,Float64,1)
    labels = get_face_labeling(model)
    V0 = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
    V = TestFESpace(model,reffe_u,conformity=:H1)
    U0 = TrialFESpace(V0, 0.0)
    U = TrialFESpace(V)

    ZU = MultiFieldFESpace([U0, U, U0])
    ZV = MultiFieldFESpace([V0, V, V0])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)
    a((u,c,p),(w,v,q)) = ∫(
        ∇(u) ⋅ ∇(q) - c ⋅ q
        + (u-ud) ⋅ w + ∇(w) ⋅ ∇(p)
        + β * (c ⋅ v) - (p ⋅ v)
    )* dΩ
    j((u,c,p), (du,dc,dp), (w,v,q)) = ∫(
        ∇(du) ⋅ ∇(q) - dc ⋅ q
        + du ⋅ w + ∇(w) ⋅ ∇(dp)
        + β * (dc ⋅ v) - (dp ⋅ v)
    )* dΩ
    op = FEOperator(a, j, ZU, ZV)
    return model, labels, ZV, ZU, Ω, dΩ, (a,j), op
end

function ControlConstrainedUniform(n::Int, ud::Function, β::T, φ::Function) where T
    model, labels, V, U, Ω, dΩ, (a,j), op = OptimalControlUniformSetup(n, ud, β)
    V0, V1 = V.spaces[1], V.spaces[2]
    ubc = interpolate_everywhere(φ, V1)
    ub = [1e10*ones(V0.nfree);ubc.free_values;1e10*ones(V0.nfree)]
    lb = -1e10*ones(num_free_dofs(V))
    return NonSymmetricObstacleProblem{T}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end

function StateConstrainedUniform(n::Int, ud::Function, β::T, φ::Function) where T
    model, labels, V, U, Ω, dΩ, (a,j), op = OptimalControlUniformSetup(n, ud, β)
    V0, V1 = V.spaces[1], V.spaces[2]
    ubu = interpolate_everywhere(φ, V0)
    ub = [ubu.free_values;1e10*ones(V1.nfree);1e10*ones(V0.nfree)]
    lb = -1e10*ones(num_free_dofs(V))
    return NonSymmetricObstacleProblem{T}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end

function Control_H1_Uniform_Setup(n::Int, ud::Function, β::T) where T

    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)

    reffe_u = ReferenceFE(lagrangian,Float64,1)
    labels = get_face_labeling(model)
    V0 = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
    V = TestFESpace(model,reffe_u,conformity=:H1)
    U0 = TrialFESpace(V0, 0.0)
    U = TrialFESpace(V)

    ZU = MultiFieldFESpace([U0, U0, U0])
    ZV = MultiFieldFESpace([V0, V0, V0])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)
    a((u,c,p),(w,v,q)) = ∫(
        ∇(u) ⋅ ∇(q) - c ⋅ q
        + (u-ud) ⋅ w + ∇(w) ⋅ ∇(p)
        + β* (∇(c) ⋅ ∇(v)) - (p ⋅ v)
    )* dΩ
    j((u,c,p), (du,dc,dp), (w,v,q)) = ∫(
        ∇(du) ⋅ ∇(q) - dc ⋅ q
        + du ⋅ w + ∇(w) ⋅ ∇(dp)
        + β* (∇(dc) ⋅ ∇(v)) - (dp ⋅ v)
    )* dΩ
    op = FEOperator(a, j, ZU, ZV)
    return model, labels, ZV, ZU, Ω, dΩ, (a,j), op
end

function ControlH1ConstrainedUniform(n::Int, ud::Function, β::T, φ::Function) where T
    model, labels, V, U, Ω, dΩ, (a,j), op = Control_H1_Uniform_Setup(n, ud, β)
    V0, V1 = V.spaces[1], V.spaces[2]
    ubc = interpolate_everywhere(φ, V1)
    ub = [1e10*ones(V0.nfree);ubc.free_values;1e10*ones(V0.nfree)]
    lb = -1e10*ones(num_free_dofs(V))
    return NonSymmetricObstacleProblem{T}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end

function State_H1_Uniform_Setup(n::Int, ud::Function, β::T) where T

    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)

    reffe_u = ReferenceFE(lagrangian,Float64,1)
    labels = get_face_labeling(model)
    V0 = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
    V = TestFESpace(model,reffe_u,conformity=:H1)
    U0 = TrialFESpace(V0, 0.0)
    U = TrialFESpace(V)

    ZU = MultiFieldFESpace([U0, U, U0])
    ZV = MultiFieldFESpace([V0, V, V0])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)
    a((u,c,p),(w,v,q)) = ∫(
        ∇(u) ⋅ ∇(q) - c ⋅ q
        + (u-ud) ⋅ w + ∇(u) ⋅ ∇(w) + ∇(w) ⋅ ∇(p)
        + β* (c ⋅ v) - (p ⋅ v)
    )* dΩ
    j((u,c,p), (du,dc,dp), (w,v,q)) = ∫(
        ∇(du) ⋅ ∇(q) - dc ⋅ q
        + du ⋅ w + ∇(du) ⋅ ∇(w) + ∇(w) ⋅ ∇(dp)
        + β* ((dc) ⋅ (v)) - (dp ⋅ v)
    )* dΩ
    op = FEOperator(a, j, ZU, ZV)
    return model, labels, ZV, ZU, Ω, dΩ, (a,j), op
end

function StateH1ConstrainedUniform(n::Int, ud::Function, β::T, φ::Function) where T
    model, labels, V, U, Ω, dΩ, (a,j), op = State_H1_Uniform_Setup(n, ud, β)
    V0, V1 = V.spaces[1], V.spaces[2]
    ubu = interpolate_everywhere(φ, V0)
    ub = [ubu.free_values;1e10*ones(V1.nfree);1e10*ones(V0.nfree)]
    lb = -1e10*ones(num_free_dofs(V))
    return NonSymmetricObstacleProblem{T}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end