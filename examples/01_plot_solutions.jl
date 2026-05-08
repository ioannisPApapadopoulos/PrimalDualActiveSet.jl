using PrimalDualActiveSet, Gridap
using Plots, LaTeXStrings

"""
Plot solutions of obstacle and thin obstacle problem.
"""
## 1D Problem 1
function sol(x)
    x₀ = 1/sqrt(10)
    if x < x₀
        return -10*x^2 + 20*x₀*x
    elseif x₀ ≤ x < 1-x₀
        return 1
    else
        return -10*x^2 + 20*(1-x₀)*x + 10 - 20*(1-x₀)
    end
end

xx = range(0,1,401)
Plots.plot(xx,ones(length(xx)), linestyle=:dash, color=:black, label="")
Plots.plot!(xx, sol.(xx),
    color=theme_palette(:auto)[1],
    linewidth=2,
    legend=:none,
    ylabel=L"u(x)", xlabel=L"x",
    xlim=[0,1],ylim=[0,1.05],
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=18,ylabelfontsize=18,legendfontsize=18)
Plots.savefig("1d-sol.pdf")

## 2D Problem 1
f(x) = 20
φ(x) = 1.0
P = ObstacleProblemUniform(2^7, f, φ, d=2)
u0 = FEFunction(P.V, zeros(P.V.nfree))

uh, iter = hik(P, u0, max_iter=1000, tol=1e-6, history=false)
writevtk(P.Ω, "obstacle_2d.vtu", cellfields=["u" => uh])
writevtk(P.Ω, "obstacle_upper_bound.vtu", cellfields=["u" => FEFunction(P.V, ones(P.V.nfree))])

## 2D Problem 2
f(x) = 20
function φ(x)
    if x[1] ≈ 0.5
        return 1.0
    else
        return 1e10
    end
end
P = ObstacleProblemUniform(2^7,f,φ,d=2)
u0 = FEFunction(P.V, zeros(P.V.nfree))
uh, iter = hik(P, u0, max_iter=1000, tol=1e-6, history=false)
writevtk(P.Ω, "obstacle_thin_2d.vtu", cellfields=["u" => uh])

## 2D Problem 3

f(x) = VectorValue(0.0,10.0)
P = SignoriniRectangle(200,40,f)
u0 = FEFunction(P.V, zeros(P.V.nfree))
uh, iter = hik(P, u0, max_iter=150, history=false)
writevtk(P.Ω, "signorini_2d.vtu", cellfields=["u" => uh])

domain = (0,5,0,0.1)
partition = (400,40)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64}, 1)
Vu = TestFESpace(model,reffe, conformity=:H1)
ub_vtk = interpolate_everywhere(x->VectorValue(0,1+0.5), Vu)
writevtk(ub_vtk.fe_space.fe_basis.trian, "bounds_2d_ub" * ".vtu", cellfields=["ub" => ub_vtk])  

## 3D Problem 3

f(x) = VectorValue(0.0,0.0,10.0)
P = SignoriniBox(50,10,10,f)
u0 = FEFunction(P.V, zeros(P.V.nfree))
uh, iter = hik(P, u0, max_iter=150, history=false)
writevtk(P.Ω, "signorini_3d.vtu", cellfields=["u" => uh])

domain = (0,5,0,1,0,0.1)
partition = (400,40)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64}, 1)
Vu = TestFESpace(model,reffe, conformity=:H1)
ub_vtk = interpolate_everywhere(x->VectorValue(0,0,1+0.5), Vu)
writevtk(ub_vtk.fe_space.fe_basis.trian, "bounds_3d_ub" * ".vtu", cellfields=["ub" => ub_vtk])  
