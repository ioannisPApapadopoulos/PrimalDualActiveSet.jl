using PrimalDualActiveSet
using Gridap, LinearAlgebra
using Plots, LaTeXStrings

function pdas_scalar_signorini_solver(nx::Integer,ny::Integer,f::Function, history::Bool=false)
    P = ScalarSignoriniRectangle(nx,ny,f)
    u0 = FEFunction(P.V, zeros(P.V.nfree))


    a(u,v) = ∫(∇(u) ⊙ ∇(v) + u ⋅ v)*P.dΩ
    j(u,du,v) = ∫(∇(du) ⊙ ∇(v) + du ⋅ v)*P.dΩ
    op = FEOperator(a, j, P.U, P.V)
    AH1 = Gridap.Algebra.jacobian(op, zeros(num_free_dofs(P.V)))

    A = Gridap.Algebra.jacobian(P.op, zeros(num_free_dofs(P.V)))


    # a(u,v) = ∫(u ⋅ v)*P.dΩ
    # j(u,du,v) = ∫(du ⋅ v)*P.dΩ
    # op = FEOperator(a, j, P.U, P.V)
    # M = Gridap.Algebra.jacobian(op, zeros(num_free_dofs(P.V)))
    uh, iter = hik(P, u0, max_iter=150, history=history)
    return uh, iter, AH1, A
end

f(x) = 10.0

iters, uniform_iters = [], []
uhs, λs, AH1s, As = [], [], [], []
uniform_uhs, uniform_λs, uniform_AH1s, uniform_As = [], [], [], []
for nx in 2 .^(4:8)
#   uh, iter, AH1, A = pdas_signorini_solver(5*nx,5,f,true)
#   push!(iters, iter[1])
#   push!(uhs, uh)
#   push!(λs, iter[3])
#   push!(As, A)

  uh, iter, AH1, A = pdas_scalar_signorini_solver(nx,nx,f,true)
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
    λ = uniform_λs[i]
    push!(uniform_norms,[])
    for j in 1:lastindex(vh)-1
        d = vh[j].free_values-vh[end].free_values
        d2 = λ[j] - λ[end]
        # push!(norms[i],sqrt(d' * As[i] * d  + d2' * (As[i] \ d2)))
        push!(uniform_norms[i],sqrt(d' * uniform_As[i] * d + d2' * (uniform_As[i] \ d2) ))
        # push!(uniform_norms[i],sqrt(d' * uniform_AH1s[i] * d ))
        # push!(norms[i], norm(d))
    end
end

Plots.plot(uniform_norms,
    labels=[L"2^{4}" L"2^{5}" L"2^{6}" L"2^{7}" L"2^{8}" L"2^{9}" L"2^{10}"],
    xlabel="HIK Iterations",
    ylabel=L"$(\Vert u^k - u^{\!\!*} \Vert^2_{A} + \Vert \lambda^k - \lambda^{\!\!*}\Vert^2_{A^{-1}})^{1/2}$",
    yaxis=:log10, 
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)



rm("tmp_nl", recursive=true)
if !isdir("tmp_nl")
    mkdir("tmp_nl")
end
level = 5;
Ω_m = uniform_uhs[level][1].cell_field.trian
createpvd("scalar_signorini") do pvd
    pvd[1] = createvtk(Ω_m, "tmp_nl/scalar_signorini_1" * ".vtu", cellfields=["u" => uniform_uhs[level][1]])
    for k in 2:length(uniform_uhs[level])
      pvd[k] = createvtk(Ω_m, "tmp_nl/scalar_signorini_$k" * ".vtu", cellfields=["u" => uniform_uhs[level][k]])
    end
end