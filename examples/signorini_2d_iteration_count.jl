using PrimalDualActiveSet
using Gridap, LinearAlgebra
using Plots, LaTeXStrings

function pdas_signorini_solver(nx::Integer,ny::Integer,f::Function, history::Bool=false)
  P = SignoriniRectangle(nx,ny,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)
  
  a(u,v) = ∫(∇(u) ⊙ ∇(v) + u ⋅ v)*P.dΩ
  j(u,du,v) = ∫(∇(du) ⊙ ∇(v) + du ⋅ v)*P.dΩ
  op = FEOperator(a, j, P.U, P.V)
  AH1 = Gridap.Algebra.jacobian(op, zeros(num_free_dofs(P.V)))

  A = Gridap.Algebra.jacobian(P.op, zeros(num_free_dofs(P.V)))
  return uh,iter, AH1, A
end

f(x) = VectorValue(0.0,10.0)

iters, uniform_iters = [], []
uhs, λs, AH1s, As = [], [], [], []
uniform_uhs, uniform_λs, uniform_AH1s, uniform_As = [], [], [], []
for nx in [5,10,20,40,80,160,320] #
#   uh, iter, AH1, A = pdas_signorini_solver(5*nx,5,f,true)
#   push!(iters, iter[1])
#   push!(uhs, uh)
#   push!(λs, iter[3])
#   push!(As, A)

  uh, iter, AH1, A = pdas_signorini_solver(5*nx,nx,f,true)
  push!(uniform_iters, iter[1])
  push!(uniform_uhs, uh)
  push!(uniform_λs, iter[3])
  push!(uniform_AH1s, AH1)
  push!(uniform_As, A)
end
uniform_iters
iters

norms = []
for i in 1:lastindex(uhs)
    vh = uhs[i]
    # λ = λs[i]
    push!(norms,[])
    for j in 1:lastindex(vh)-1
        d = vh[j].free_values-vh[end].free_values
        # d2 = λ[j] - λ[end]
        # push!(norms[i],sqrt(d' * As[i] * d  + d2' * (As[i] \ d2)))
        push!(norms[i],sqrt(d' * As[i] * d  ))
        # push!(norms[i], norm(d))
    end
end
uniform_norms = []
for i in 1:lastindex(uniform_uhs)
    vh = uniform_uhs[i]
    # λ = uniform_λs[i]
    push!(uniform_norms,[])
    for j in 1:lastindex(vh)-1
        d = vh[j].free_values-vh[end].free_values
        # d2 = λ[j] - λ[end]
        # push!(norms[i],sqrt(d' * As[i] * d  + d2' * (As[i] \ d2)))
        # push!(uniform_norms[i],sqrt(d' * uniform_As[i] * d + d2' * (uniform_As[i] \ d2) ))
        push!(uniform_norms[i],sqrt(d' * uniform_AH1s[i] * d ))
        # push!(norms[i], norm(d))
    end
end

Plots.plot(uniform_norms,
    labels=[L"5" L"10" L"20" L"40" L"80" L"160" L"320"],
    xlabel="HIK Iterations",
    ylabel=L"$(\Vert u^k - u^{\!\!*} \Vert^2_{A} + \Vert \lambda^k - \lambda^{\!\!*}\Vert^2_{A^{-1}})^{1/2}$",
    yaxis=:log10, 
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)
Plots.savefig("HIK-convergence-uniform-signorini.pdf")

Plots.plot(uniform_norms,
    labels=[L"5" L"10" L"20" L"40" L"80" L"160" L"320"],
    xlabel="HIK Iterations",
    ylabel=L"$\Vert u^k - u^{\!\!*} \Vert_{H^1(\Omega)}$",
    yaxis=:log10, 
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)
Plots.savefig("HIK-convergence-uniform-signorini-H1.pdf")


Plots.plot(norms,
    labels=[L"5" L"10" L"20" L"40" L"80" L"160" L"320"],
    xlabel="HIK Iterations",
    ylabel=L"$(\Vert u^k - u^{\!\!*} \Vert^2_{A} + \Vert \lambda^k - \lambda^{\!\!*}\Vert^2_{A^{-1}})^{1/2}$",
    yaxis=:log10, 
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)


eoc = []
sls = []
for i in 1:lastindex(uhs)
    n = uniform_norms[i]
    push!(eoc, [])
    for j in 1:lastindex(n)-2
        push!(eoc[i], log(n[j+2]/n[j+1])/log(n[j+1]/n[j]))
    end
    # push!(sls,findall(x->x>1.01, eoc[end])[1])
end



nx = 100
ny = 20
@time uhs, iter_actives = pdas_signorini_solver(nx,ny,true,f);
Ω = uhs[1].fe_space.fe_basis.trian
createpvd("signorini_2d_$ny") do pvd
    pvd[1] = createvtk(Ω, "tmp_nl/signorini_2d_$(ny)_1" * ".vtu", cellfields=["u" => uhs[1]])
    for k in 2:length(uhs)
      pvd[k] = createvtk(Ω, "tmp_nl/signorini_2d_$(ny)_$k" * ".vtu", cellfields=["u" => uhs[k]])
    end
end

domain = (0,5,0,0.1)
partition = (400,40)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64}, 1)
Vu = TestFESpace(model,reffe, conformity=:H1)
ub_vtk = interpolate_everywhere(x->VectorValue(0,1+0.5), Vu)
writevtk(ub_vtk.fe_space.fe_basis.trian, "bounds_ub" * ".vtu", cellfields=["ub" => ub_vtk])  