using PrimalDualActiveSet, Gridap, LinearAlgebra

"""
A (now removed) hyperelasticity example with iteration counts and convergence plot
"""
function pdas_hypercontact_solver(nx::Integer,ny::Integer,f::Function, history::Bool=false)
    P = HyperContactRectangle(nx,ny,f)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    uh, iter = hik(P, u0, max_iter=150, history=history)

    a(u,v) = ∫(∇(u) ⊙ ∇(v) + u ⋅ v)*P.dΩ
    j(u,du,v) = ∫(∇(du) ⊙ ∇(v) + du ⋅ v)*P.dΩ
    op = FEOperator(a, j, P.U, P.V)
    A = Gridap.Algebra.jacobian(op, zeros(num_free_dofs(P.V)))
    return uh,iter,A
end

f(x) = VectorValue(0.0,10.0)

iters = []
uhs, λs, As = [], [], []
for nx in [5,10,20,40,80,160,320] 
  uh, iter,A = pdas_hypercontact_solver(5*nx,nx,f,true)
  push!(iters, iter[1])
  push!(uhs, uh)
  push!(λs, iter[3])
  push!(As,A)
end
iters

# level=2
# Ω = uhs[level][1].fe_space.fe_basis.trian
# createpvd("hypercontact") do pvd
#     pvd[1] = createvtk(Ω, "hypercontact_1" * ".vtu", cellfields=["u" => uhs[level][1]])
#     for k in 2:length(uhs[level])
#       pvd[k] = createvtk(Ω, "hypercontact_$k" * ".vtu", cellfields=["u" => uhs[level][k]])
#     end
# end


norms = []
for i in 1:lastindex(uhs)
    vh = uhs[i]
    push!(norms,[])
    for j in 1:lastindex(vh)-1
        d = vh[j].free_values-vh[end].free_values
        push!(norms[i],sqrt(d' * As[i] * d  ))
    end
end

Plots.plot(norms,
    labels=[L"5" L"10" L"20" L"40" L"80" L"160" L"320"],
    xlabel="HIK Iterations",
    ylabel=L"$\Vert u^k_h - u^{\!\!*}_h \Vert_{H^1(\Omega)}$",
    yaxis=:log10, 
    yticks=[1e-10,1e-8,1e-6,1e-4,1e-2,1e0],
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)
Plots.savefig("hyperelastic_convergence.pdf")