using PrimalDualActiveSet
using Gridap, LinearAlgebra

f(x) = VectorValue(0.0,0.0,10.0)
function pdas_signorini_solver(n::Integer,f::Function, history::Bool=false)
  print("n=$n\n")
  P = SignoriniBox(5*n,n,n,f)
  print("Dofs: $(P.V.nfree)\n")
  # nullsp = linear_elasticity_nullsp(P, 3)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, tol=1e-5, solver_flag=Val(1), history=history)#, nullsp=nullsp)

  return uh, iter
end

uh, iter = pdas_signorini_solver(160,f)

# uniform_iters = [], []
# for n in [160] #
#   uh, iter = pdas_signorini_solver(n,f)
#   push!(uniform_iters, iter[1])
# end
# uniform_iters