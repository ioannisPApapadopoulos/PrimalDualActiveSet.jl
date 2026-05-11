using PrimalDualActiveSet, Gridap, LinearAlgebra
using Plots, LaTeXStrings

# 2D Problem 3 Iteration Counts
function pdas_signorini_solver(nx::Integer,ny::Integer,f::Function, history::Bool=false)
  P = SignoriniRectangle(nx,ny,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)

  A = Gridap.Algebra.jacobian(P.op, zeros(num_free_dofs(P.V)))
  return uh,iter,A
end
f(x) = VectorValue(0.0,10.0)

iters, uhs, λs, As = [], [], [], []
for nx in [5,10,20,40,80,160,320]
  uh, iter, A = pdas_signorini_solver(5*nx,nx,f,true)
  push!(iters, iter[1])
  push!(uhs, uh)
  push!(λs, iter[3])
  push!(As, A)
end
print("HIK 2D Iteration Counts: $(iters)")

# 2D Problem 3 Convergence Plot
norms = []
for i in 1:lastindex(uhs)
    vh = uhs[i]
    λ = λs[i]
    push!(norms,[])
    for j in 1:lastindex(vh)-1
        d = vh[j].free_values-vh[end].free_values
        d2 = λ[j] - λ[end]
        push!(norms[i],sqrt(d' * As[i] * d  + d2' * (As[i] \ d2)))
    end
end

Plots.plot(norms,
    labels=[L"5" L"10" L"20" L"40" L"80" L"160" L"320"],
    xlabel="HIK Iterations",
    ylabel=L"$(\Vert u^k - u^{\!\!*} \Vert^2_{A} + \Vert \lambda^k - \lambda^{\!\!*}\Vert^2_{A^{-1}})^{1/2}$",
    yaxis=:log10, 
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)
Plots.savefig("HIK-convergence-uniform-signorini.pdf")

# 3D Problem 3 Iteration Counts
function pdas_signorini_solve_3d(nx::Integer,ny::Integer,nz::Integer,f::Function, history::Bool=false)
  P = SignoriniBox(nx,ny,nz,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)
  return uh,iter
end
f(x) = VectorValue(0.0,0.0,10.0)

iters, uhs, λs, As = [], [], [], []
for nx in [5,10,20,40,80]
  uh, iter, A = pdas_signorini_solver_3d(5*nx,nx,nx,f)
  push!(iters, iter[1])
  push!(uhs, uh)
end
print("HIK 3D Iteration Counts: $(iters)")