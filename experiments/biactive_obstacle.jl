using PrimalDualActiveSet, Gridap

function f(x)
    if x[1] <= 0
        return 0.0
    else
        return 12*x[1]^2
    end
end
φ(x) = 0.0
function g(x)
    if x[1] <= 0
        return 0.0
    else
        return -x[1]^4
    end
end

iters = Int[]
for n in 2 .^(4:10)
    P = BiactiveObstacle(n,f,φ,g)
    u0 = interpolate_everywhere(x->0.0, P.U)
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(iters, iter[1])
end

vhs = [vh];
level = 1;
Ω_m = vhs[level][1].cell_field.trian
createpvd("biactive_obstacle") do pvd
    pvd[1] = createvtk(Ω_m, "tmp_nl/biactive_obstacle_1" * ".vtu", cellfields=["u" => vhs[level][1]])
    for k in 2:length(vhs[level])
      pvd[k] = createvtk(Ω_m, "tmp_nl/biactive_obstacle_$k" * ".vtu", cellfields=["u" => vhs[level][k]])
    end
end



function BiactiveObstacle(n::Int, f::Function, φ::Function, g::Function)
    domain = (-1,1,-1,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)
    model = simplexify(model)
    reffe_u = ReferenceFE(lagrangian,Float64,1)
    labels = get_face_labeling(model)
    V = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)

    U = TrialFESpace(V, g)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,6)
    a(u, v) =∫(∇(u) ⋅ ∇(v) - f ⋅ v) * dΩ
    j(u, du, v) =∫(∇(du) ⋅ ∇(v)) * dΩ
    op = FEOperator(a, j, U, V)

    # ub = 1e10*ones(V.nfree)
    lb = -1e10*ones(V.nfree)
    ub = interpolate_everywhere(φ, V).free_values
    return ObstacleProblem{Float64}(model, labels, V, U, Ω, dΩ, (a,j), op, lb, ub)
end


φ(x) = 0.0
function g(x)
    if x[1]^2 + x[2]^2 < 0.25
        return -(1-4*x[1]^2-4*x[2]^2)^4
    else
        return 0.0
    end
end

gh = interpolate_everywhere(x->g(x), P.U)
function d2g(x)
    if x[1]^2 + x[2]^2 < 0.25
        return -(32*(-4*x[1]^2-4*x[2]^2+1)^2*(28*x[1]^2+4*x[2]^2-1) + 32*(-4*x[2]^2-4*x[1]^2+1)^2*(28*x[2]^2+4*x[1]^2-1))
    else
        return 0.0
    end
end
function f(x)
    if x[1]^2 + x[2]^2 > 0.75
        return -d2g(x) + 1.0
    else
        return -d2g(x)
    end
end
# fh = interpolate_everywhere(x->f(x), P.U)

iters = Int[]
for n in 2 .^(4:7)
# n = 2^5
    P = BiactiveObstacle(n,f,φ,g)
    u0 = interpolate_everywhere(x->0.0, P.U)
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(iters, iter[1])
end


vhs = [[u0]];
level = 1;
Ω_m = vhs[level][1].cell_field.trian
createpvd("biactive_obstacle") do pvd
    pvd[1] = createvtk(Ω_m, "tmp_nl/biactive_obstacle_1" * ".vtu", cellfields=["u" => vhs[level][1]])
    for k in 2:length(vhs[level])
      pvd[k] = createvtk(Ω_m, "tmp_nl/biactive_obstacle_$k" * ".vtu", cellfields=["u" => vhs[level][k]])
    end
end

a(u, v) =∫(∇(u) ⋅ ∇(v) - f ⋅ v) * dΩ
j(u, du, v) =∫((du) ⋅ (v)) * dΩ
op = FEOperator(a, j, P.U, P.V)

M = Gridap.Algebra.jacobian(op, u0)

u0.free_values .= iter[3][end]