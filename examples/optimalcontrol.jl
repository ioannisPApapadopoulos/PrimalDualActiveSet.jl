using PrimalDualActiveSet, Gridap
using Plots, LaTeXStrings

domain = (0,1,0,1)
n=50
β = 1e-2
ud(x) = 5.0
φ(x) = 1.0

control_iters = []
for n in [5,10,20,40,80,160,320]
    PC = ControlConstrainedUniform(n, ud, β, φ)
    z0 = FEFunction(PC.V, zeros(num_free_dofs(PC.V)))
    zh, iter = hik(PC, z0)
    push!(control_iters, iter)
end

control_h1_iters = []
for n in [5,10,20,40,80,160,320]
    PC = ControlH1ConstrainedUniform(n, ud, β, φ)
    z0 = FEFunction(PC.V, zeros(num_free_dofs(PC.V)))
    zh, iter = hik(PC, z0)
    push!(control_h1_iters, iter)
end

state_iters = []
for n in [5,10,20,40,80,160,320]
    PS = StateConstrainedUniform(n, ud, β, φ)
    z0 = FEFunction(PS.V, zeros(num_free_dofs(PS.V)))
    zh, iter = hik(PS, z0)
    push!(state_iters, iter)
end


state_h1_iters = []
φ(x) = 0.2
for n in [5,10,20,40,80,160,320]
    PC = StateH1ConstrainedUniform(n, ud, β, φ)
    z0 = FEFunction(PC.V, zeros(num_free_dofs(PC.V)))
    zh, iter = hik(PC, z0)
    push!(state_h1_iters, iter)
end

xx = range(0,1,50)
Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03

yh, uh, ph = zh.single_fe_functions
p = surface(xx, xx, (x, y) -> yh(Point.(x,y)), 
    color=:redsblues, #:vik,
    xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y)",
    # camera=(30,-30),
    margin=(-6, :mm),
)

