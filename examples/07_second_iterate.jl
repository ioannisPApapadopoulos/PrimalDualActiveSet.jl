using PrimalDualActiveSet, Gridap

f(x) = 20
φ(x) = 1.0
n = 2^10
P = ObstacleProblemUniform(n, f, φ, d=2)
u0 = FEFunction(P.V, zeros(P.V.nfree))
vh, iter = hik(P, u0, max_iter=2, history=true)
writevtk(vh[2].fe_space.fe_basis.trian, "obstacle_second_iterate.vtu", cellfields=["u" => vh[2]])  

f(x) = 20
function φ(x)
    if x[1] ≈ 0.5
        return 1.0
    else
        return 1e10
    end
end
P = ObstacleProblemUniform(n, f, φ, d=2)
u0 = FEFunction(P.V, zeros(P.V.nfree))
vh, iter = hik(P, u0, max_iter=2, history=true)
writevtk(vh[2].fe_space.fe_basis.trian, "thin_obstacle_second_iterate.vtu", cellfields=["u" => vh[2]])  