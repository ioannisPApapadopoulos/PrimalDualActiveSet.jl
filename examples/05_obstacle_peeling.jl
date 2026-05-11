using PrimalDualActiveSet, Gridap, LinearAlgebra
using Plots, LaTeXStrings

# 1D Obstacle Problem Peeling
f(x) = 20
φ(x) = 1.0
hik_its, vhs = [], []
ns = 2 .^(4:5)
for n in ns
    P = ObstacleProblemUniform(n, f, φ)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(hik_its, iter)
    push!(vhs, vh)
end

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

# 2D Obstacle Problem Peeling
hik_its_2d  = []
vhs = []
ns = 2 .^(4:5)
for n in ns
    P = ObstacleProblemUniform(n,f,φ,d=2)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(hik_its_2d, iter)
    push!(vhs,vh)
end

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