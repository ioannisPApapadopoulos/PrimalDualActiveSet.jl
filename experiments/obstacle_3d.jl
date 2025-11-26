using PrimalDualActiveSet
using Gridap, LinearAlgebra

f(x) = 20
φ(x) = 1.0

hik_its = Integer[]
dofs = Integer[]
ns = 2 .^(7:7) #:10
for n in ns
    P = ObstacleProblemUniform(n,f,φ,d=3)
    push!(dofs, P.V.nfree)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P, u0, max_iter=1000, tol=1e-6, solver_flag=Val(1))
    push!(hik_its, iter[1])
end
print("HIK 3D Dofs: $(dofs), Iteration Counts: $(hik_its)")