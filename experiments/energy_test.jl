using PrimalDualActiveSet
using Plots

u(x) = -5.0*(x-1.0)*x

xx = range(0,1,101)
Plots.plot(xx,u.(xx))

function pu(x)
    ux = u(x)
    if ux > 1
        return 1.0
    else
        return ux
    end
end

Plots.plot!(xx,pu.(xx))


using Gridap
f(x) = 10
φ(x) = 1.0
n = 5
P = ObstacleProblemUniform(n, f, φ)

u0 = FEFunction(P.V, zeros(P.V.nfree))
vh, iter = hik(P, u0, max_iter=2, history=true)

Plots.plot!(vh[2](Point.(xx)))


domain = (0,1)
partition = (n,)
reffe_u = ReferenceFE(lagrangian,Float64,1)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
V = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
U = TrialFESpace(V, 0.0)
Ω = Triangulation(model)
dΩ = Measure(Ω,6)

puh = interpolate_everywhere(x->pu(x[1]), V)
a(u, v) =∫(∇(u) ⋅ ∇(v) - ∇(puh) ⋅ ∇(v)) * dΩ
j(u, du, v) =∫(∇(du) ⋅ ∇(v)) * dΩ
op = FEOperator(a, j, U, V)
u0 = FEFunction(V, zeros(V.nfree))
r, A = Gridap.Algebra.residual_and_jacobian(op,u0)
uh = FEFunction(V,-A \ r)

Plots.plot!(uh(Point.(xx)))

norm(uh.free_values - vh.free_values)