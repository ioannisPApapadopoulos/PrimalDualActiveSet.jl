function _2d_top_dofs(x, ub::T) where T
    if x[2] ≈ 1.0
        return VectorValue(1e10, ub)
    else
        return VectorValue(1e10, 1e10)
    end
end

function _2d_scalar_top_dofs(x, ub::T) where T
    if x[2] ≈ 1.0
        return ub
    else
        return 1e10
    end
end
function _3d_top_dofs(x, ub::T) where T
    if x[3] ≈ 1.0
        return VectorValue(1e10, 1e10, ub)
    else
        return VectorValue(1e10, 1e10, 1e10)
    end
end

function SignoriniRectangle(nx::Integer, ny::Integer, f::Function)
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

    E = 200.0
    ν = 0.3
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)

    a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) - f ⊙ v)*dΩ
    j(u,du,v) = ∫( ε(v) ⊙ (σ∘ε(du)) )*dΩ
    
    op = FEOperator(a, j, U, V)

    # Γ = BoundaryTriangulation(Ω, tags=[3,6,4]) # upper bound
    # dΓ = Measure(Γ,5)

    # a_b(u,v) = ∫(VectorValue(0,1.0) ⋅ v)*dΓ
    # op_b = FEOperator(a_b, U, V)

    # ub = Gridap.Algebra.residual(op_b, FEFunction(V, zeros(V.nfree)))
    # ub[abs.(ub) .> 0.0] .= 0.5
    # ub[ub .== 0.0] .= 1e10

    ub = interpolate_everywhere(x->_2d_top_dofs(x, 0.5), V).free_values

    lb = -1e10*ones(V.nfree)
    return ObstacleProblem{Float64}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end

function ScalarSignoriniRectangle(nx::Integer, ny::Integer, f::Function)
    domain = (0,1,0,1)
    partition = (nx,ny)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)
    reffe = ReferenceFE(lagrangian,Float64, 1)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"left",[1,3,7])
    add_tag_from_tags!(labels,"right",[2,4,8])
    add_tag_from_tags!(labels,"bottom",[5])

    V = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["left", "right", "bottom"])

    U = TrialFESpace(V, 0.0)

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)

    a(u, v) =∫(∇(u) ⋅ ∇(v) - f ⋅ v) * dΩ
    j(u, du, v) =∫(∇(du) ⋅ ∇(v)) * dΩ
    op = FEOperator(a, j, U, V)

    # Γ = BoundaryTriangulation(Ω, tags=[3,6,4]) # upper bound
    # dΓ = Measure(Γ,5)

    # a_b(u,v) = ∫(1.0 ⋅ v)*dΓ
    # op_b = FEOperator(a_b, U, V)

    # ub = Gridap.Algebra.residual(op_b, FEFunction(V, zeros(V.nfree)))
    # ub[abs.(ub) .> 0.0] .= 0.5
    # ub[ub .== 0.0] .= 1e10

    ub = interpolate_everywhere(x->_2d_scalar_top_dofs(x, 1.0), V).free_values
    lb = -1e10*ones(V.nfree)
    return ObstacleProblem{Float64}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end


"""
Box Labelling:
vertices: 1 - (0,0,0), 2 - (1,0,0), 3 - (0,1,0), 4 - (1,1,0), 5 - (0,0,1), 6 - (1,0,1), 7- (0,1,1), 8 - (1,1,1)
edge x direction: 9 - y=z=0, 10 - z=0,y=1, 11-z=1,y=0, 12-y=z=1
edge y direction 13 - x=z=0, 14 - x-1, z=0, 15-z=1, x=0, 16-x=z=1
edge z direction 17- x=y=0, 18-x=1,y=0 19-x=0,y=1,  20-x=y=1
faces: 21 - bottom, 22-top, 23-facing us, 24-facing away, 25-left, 26-right

"""
function SignoriniBox(nx::Integer,ny::Integer,nz::Integer,f::Function)

    domain = (0,5,0,1,0,1)
    partition = (nx,ny,nz)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)
    reffe = ReferenceFE(lagrangian,VectorValue{3,Float64}, 1)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"left-right",[1,2,3,4,5,6,7,8,17,18,19,20,13,14,15,16,25,26])
    add_tag_from_tags!(labels,"contact", [22,5,6,7,8,11,12,15,16])
    V = TestFESpace(model,reffe, conformity=:H1, dirichlet_tags=["left-right"])

    g(x) = VectorValue(0.0,0.0,0.0)

    U = TrialFESpace(V,[g])

    E = 200.0
    ν = 0.3
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

    Ω = Triangulation(model)
    dΩ = Measure(Ω,5)

    a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) - f ⊙ v)*dΩ
    j(u,du,v) = ∫( ε(v) ⊙ (σ∘ε(du)) )*dΩ

    op = FEOperator(a, j, U, V)

    # Γ = BoundaryTriangulation(Ω, tags=["contact"])
    # dΓ = Measure(Γ,5)

    # a_b(u,v) = ∫(VectorValue(0,0,1.0) ⋅ v)*dΓ
    # op_b = FEOperator(a_b, U, V)

    # ub = Gridap.Algebra.residual(op_b, FEFunction(V, zeros(V.nfree)))
    # ub[abs.(ub) .> 0.0] .= 0.5
    # ub[ub .== 0.0] .= 1e10

    ub = interpolate_everywhere(x->_3d_top_dofs(x, 0.5), V).free_values
    lb = -1e10*ones(V.nfree)
    return ObstacleProblem{Float64}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end