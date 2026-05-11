using PrimalDualActiveSet, Gridap, LinearAlgebra
using Plots, LaTeXStrings

f(x) = 20
φ(x) = 1.0
hik_its, vhs = [], []
ns = [2^7, 2^11]
for n in ns
    P = ObstacleProblemUniform(n, f, φ)
    u0 = FEFunction(P.V, zeros(P.V.nfree))
    vh, iter = hik(P, u0, max_iter=1000, history=true)
    push!(hik_its, iter)
    push!(vhs, vh)
end

mkdir("video")
for k in 1:2
    vh = vhs[k]
    vals = []
    xx = range(0,1,ns[k]+1)
    for i in 1:length(vh)
        push!(vals, vh[i](Point.(xx,)))
    end

    for i in 1:length(vals)
        Plots.plot(xx,vals[i], 
            linewidth=1.5,
            xlim=[0,1],ylim=[0,1.3],
            ylabel=L"u(x)", xlabel=L"x",
            legend_column=-1,
            # label=L"n=2^{11}",
            xtickfontsize=10, ytickfontsize=10,xlabelfontsize=18,ylabelfontsize=18,legendfontsize=12)
        p = Plots.plot!(xx,ones(length(xx)), linestyle=:dash, color=:black, label="")
        display(p)
        Plots.savefig("video/obstacle_peeling_n_$(ns[k])_$i.png")
    end
end

# ffmpeg -framerate 8 -i obstacle_peeling_n_2048_%d.png -pix_fmt yuv420p fine.mp4
