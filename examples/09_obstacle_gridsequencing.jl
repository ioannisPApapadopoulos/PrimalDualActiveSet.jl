using PrimalDualActiveSet, Gridap

f(x) = 20
φ(x) = 1.0
ps = 4:10
hik_iters = []

function gridsequence(f::Function,ps::UnitRange{Int})
    P = ObstacleProblemUniform(2^ps[1],f,φ,d=2)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    uh, iter = hik(P, u0, max_iter=1000)
    push!(hik_iters, iter)

    for i in 2:length(ps)
        P = ObstacleProblemUniform(2^ps[i],f,φ,d=2)

        # This is too slow
        # u0 = interpolate_everywhere(x->uh(Point(x...)), P.U)

        # This is a quicker interpolation that works since we are using P1-FEM
        Vd = TestFESpace(P.model, ReferenceFE(lagrangian,Float64,1), conformity=:H1)
        u1 = FEFunction(Vd, zeros(Vd.nfree))
        u1.free_values .= uh(P.V.fe_basis.trian.grid.node_coordinates)

        # This then removes the Dirichlet dofs
        u0 = interpolate(u1, P.V)

        uh, iter = hik(P, u0, max_iter=1000)
        push!(hik_iters, iter)
    end

    return hik_iters
end

hik_iters = gridsequence(f,ps)