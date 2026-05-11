using PrimalDualActiveSet, Gridap, LinearAlgebra

f(x) = 20
φ(x) = 1.0

hik_its = Integer[]
ns = 2 .^(4:10)
for n in ns
    P = ObstacleProblemUniform(n,f,φ,d=2)
    u0 = FEFunction(P.V, zeros(P.V.nfree))

    dΩ = Measure(P.Ω,2)
    a(u,v) = ∫(u ⋅ v)*dΩ
    j(u,du,v) = ∫(du ⋅ v)*dΩ
    op = FEOperator(a, j, P.U, P.V)
    M = Gridap.Algebra.jacobian(op, zeros(num_free_dofs(P.V)))

    vh, iter = ssn(P, M, u0, max_iter=20)
    push!(hik_its, iter)
end
print("HIK 2D Iteration Counts: $(hik_its)")