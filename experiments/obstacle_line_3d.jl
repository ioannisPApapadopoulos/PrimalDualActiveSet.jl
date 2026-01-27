using PrimalDualActiveSet
using Gridap, LinearAlgebra
using Plots, LaTeXStrings


f(x) = 20
function φ(x)
    if x[1] ≈ 0.5
        return 1.0
    else
        return 1e10
    end
end

hik_its = Integer[]
ns = 2 .^(6:7)
for n in ns
    P = ObstacleProblemUniform(n,f,φ,d=3)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P, u0, max_iter=1000, history=false)
    push!(hik_its, iter[1])
end
print("HIK 3D Iteration Counts: $(hik_its)")