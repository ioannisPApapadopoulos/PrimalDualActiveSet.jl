using PrimalDualActiveSet
using Gridap, LinearAlgebra

f(x) = 20
φ(x) = 1.0

hik_its_2d  = []
vhs = []
ns = 2 .^(4:5)
for n in ns
    P = ObstacleProblemUniform(n,f,φ,d=2)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    # vh, iter = hik(P.op, u0, P.lb, P.ub, max_iter=1000, history=true)
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(hik_its_2d, iter)
    push!(vhs,vh)
end
hik_its_2d

for k in 1:2
    step = 1/(length(vhs[k]))
    wh = FEFunction(vhs[k][1].fe_space, zeros(length(vhs[k][1].free_values)))
    wh.free_values[hik_its_2d[k][2][end]] .= 1
    for j in length(vhs[k]):-1:3
        h1 = hik_its_2d[k][2][j]
        h2 = hik_its_2d[k][2][j-1]
        diff = setdiff(h2,h1)
        print("Assign=$(step/2 + (j-2) *step), diff=$diff.\n")
        wh.free_values[diff] .= step/2 + (j-2) *step
    end

    writevtk(vhs[k][1].fe_space.fe_basis.trian, "active_$(16*k).vtu", cellfields=["active" => wh])
end