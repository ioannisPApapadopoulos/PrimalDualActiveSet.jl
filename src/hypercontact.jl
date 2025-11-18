function HyperContactRectangle(nx::Integer, ny::Integer, f::Function)
    domain = (0,5,0,1)
    partition = (nx,ny)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)
    reffe = ReferenceFE(lagrangian,VectorValue{2,Float64}, 1)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"left",[1,3,7])
    add_tag_from_tags!(labels,"right",[2,4,8])

    V = TestFESpace(model,reffe, conformity=:H1, dirichlet_tags=["left", "right"])

    g(x) = VectorValue(0.0,0.0)
    U = TrialFESpace(V,[g,g])

    E = 1e3
    ν = 0.3
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    F(∇) = one(∇) + ∇
    C(∇) = F(∇)' * F(∇)
    trC(∇) = tr(C(∇))
    J(∇) = det(F(∇))
    Ψ(∇) = μ/2*(trC(∇) - 2.0) - μ*log(J(∇)) + λ/2*log(J(∇))^2
    invFT(∇) = (inv ∘ (F∘(∇)))'

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)

    dΨ(∇u, ∇v) = μ * (tr(∇v) + ∇u ⊙ ∇v ) - μ * invFT(∇u) ⊙ ∇v  + λ * (log ∘ (J ∘ ∇u) *  invFT(∇u) ⊙ ∇v)
    d2Ψ(∇u,∇du,∇v) = μ * (∇du ⊙ ∇v) + μ * (invFT(∇u) ⋅ (∇v)' ⋅ invFT(∇u)) ⊙ ∇du  - λ * (log ∘ (J ∘ ∇u) *  (invFT(∇u) ⋅ (∇v)' ⋅ invFT(∇u)) ⊙ ∇du) + λ*((invFT(∇u) ⊙ ∇du)⋅(invFT(∇u) ⊙ ∇v))

    a(u, v) = ∫( dΨ(∇(u), ∇(v)) - v ⋅ f)*dΩ
    j(u, du, v) = ∫(d2Ψ(∇(u),∇(du), ∇(v)))dΩ
    
    op = FEOperator(a, j, U, V)

    ub = interpolate_everywhere(x->_2d_top_dofs(x, 0.2), V).free_values
    lb = -1e10*ones(V.nfree)
    return NonSymmetricObstacleProblem{Float64}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end