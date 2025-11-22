using PrimalDualActiveSet
using Gridap, LinearAlgebra
using Plots, LaTeXStrings

f(x) = VectorValue(0.0,0.0,10.0)
function pdas_signorini_solver(n::Integer,f::Function, history::Bool=false)
  P = SignoriniBox(5*n,n,n,f)
  nullsp = linear_elasticity_nullsp(P, 3)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, solver_flag=Val(1), history=history, nullsp=nullsp)

  A = Gridap.Algebra.jacobian(P.op, zeros(num_free_dofs(P.V)))
  return uh,iter, A
end


iters, uniform_iters = [], []
uniform_uhs, uniform_λs,uniform_As = [], [], []
tic = @elapsed for n in [10] #
  uh, iter, A = pdas_signorini_solver(n,f,true)
  push!(uniform_iters, iter[1])
  push!(uniform_uhs, uh)
end
uniform_iters