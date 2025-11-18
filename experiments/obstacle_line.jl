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
ns = 2 .^(4:10)
vhs, λs, As = [], [], []
for n in ns
    P = ObstacleProblemUniform(n,f,φ,d=2)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(hik_its, iter[1])
    push!(vhs, vh)
    push!(λs, iter[3])
    push!(As, Gridap.Algebra.jacobian(P.op, zeros(n)))
end
print("HIK 2D Iteration Counts: $(hik_its)")

# Plots.plot(vhs[3][end].free_values)

rm("tmp_nl", recursive=true)
if !isdir("tmp_nl")
    mkdir("tmp_nl")
end
level = 7;
Ω_m = vhs[level][1].cell_field.trian
createpvd("scalar_signorini") do pvd
    pvd[1] = createvtk(Ω_m, "tmp_nl/scalar_signorini_1" * ".vtu", cellfields=["u" => vhs[level][1]])
    for k in 2:length(vhs[level])
      pvd[k] = createvtk(Ω_m, "tmp_nl/scalar_signorini_$k" * ".vtu", cellfields=["u" => vhs[level][k]])
    end
end


norms = []
for i in 1:lastindex(vhs)
    vh = vhs[i]
    λ = λs[i]
    push!(norms,[])
    for j in 1:lastindex(vh)-1
        d = vh[j].free_values-vh[end].free_values
        d2 = λ[j] - λ[end]
        push!(norms[i],sqrt(d' * As[i] * d  + d2' * (As[i] \ d2)))
    end
end

Plots.plot(norms,
    labels=[L"2^{4}" L"2^{5}" L"2^{6}" L"2^{7}" L"2^{8}" L"2^{9}" L"2^{10}"],
    xlabel="HIK Iterations",
    ylabel=L"$(\Vert u^k - u^{\!\!*} \Vert^2_{A} + \Vert \lambda^k - \lambda^{\!\!*}\Vert^2_{A^{-1}})^{1/2}$",
    yaxis=:log10, 
    linewidth=2,
    labelfontsize=12,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=9)
Plots.savefig("obstacle-line-convergence.pdf")