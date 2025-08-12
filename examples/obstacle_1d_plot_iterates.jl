using PrimalDualActiveSet
using Gridap, LinearAlgebra
using Plots, LaTeXStrings

f(x) = 20
φ(x) = 1.0
hik_its, bm_its = [], []
vhs = []

ns = 2 .^(4:5)
for n in ns
    P = ObstacleProblemUniform(n, f, φ)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P.op, u0, P.lb, P.ub, max_iter=1000, history=true)
    push!(hik_its, iter)
    push!(vhs, vh)
end
hik_its


markers = [:circle, :rect,  :diamond, :hexagon, :utriangle,  :xcross, 
    :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, 
    :star7, :star8, :vline, :hline, :+, :x]

for k in 1:2
    vh = vhs[k]
    vals = []
    xx = range(0,1,ns[k]+1)
    for i in 1:length(vh)
        push!(vals, vh[i](Point.(xx,)))
    end
    xlabels = []
    for i in 1:4k:length(xx)
        push!(xlabels, "$(xx[i])")
        for j in 1:(4k-1)
            push!(xlabels, "")
        end
    end
    xlabels=xlabels[1:end-(2k-1)]
    p = Plots.plot(xx,ones(length(xx)), linestyle=:dash, color=:black, label="")
    for i in 1:length(vals)
        Plots.plot!(xx,vals[i], 
            # marker=:dot,
            linewidth=1.5,
            label="$i",
            legend=:bottom,
            marker=markers[i],
            xlim=[0,1],ylim=[0,1.3],
            xticks=(xx,xlabels),
            ylabel=L"u(x)", xlabel=L"x",
            legend_column=-1,
            xtickfontsize=10, ytickfontsize=10,xlabelfontsize=18,ylabelfontsize=18,legendfontsize=12)
    end
    display(p)
    Plots.savefig("1d_active_sets_$(16*k).pdf")
end