using PrimalDualActiveSet, Gridap, Gridap.Geometry, Gridap.Adaptivity

nx = 5

function adapt_fun(x, level)
    gap = 1/(level*nx)
    if x[2] ≥ 1 - gap
        return 1.0
    else
        return 0.0
    end
end

# function adapt_fun(x, level)
#     gap = 1/(level*nx)
#     if x[2] ≥ 1 - gap || x[2] ≤ gap || x[1] ≤ gap || x[1] ≥ 5 - gap
#         return 1.0
#     else
#         return 0.0
#     end
# end

model = simplexify(CartesianDiscreteModel((0,5,0,1),(5*nx,nx)))
models = Any[model]
reffe = ReferenceFE(lagrangian,Float64,1)

for level in 1:6
    V = TestFESpace(model,reffe)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)
    adapt_fun_level(x) = adapt_fun(x,level)
    ηh(u)  = ∫(adapt_fun_level)*dΩ
    uh = FEFunction(V, zeros(V.nfree))
    η = estimate(ηh,uh)
    m = DorflerMarking(0.9999)
    I = Adaptivity.mark(m,η)

    method = Adaptivity.NVBRefinement(model)
    model = Adaptivity.get_model(refine(method,model;cells_to_refine=I))
    push!(models, model)
end

Ω = Triangulation(models[7])
writevtk(Ω,"fmodel")

function pdas_signorini_solver(model,f::Function, history::Bool=false)
  P = SignoriniRectangle(model,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)

  A = Gridap.Algebra.jacobian(P.op, zeros(num_free_dofs(P.V)))
  return uh, iter, A
end

f(x) = VectorValue(0.0,10.0)

adaptive_iters = []
adaptive_uhs, adaptive_λs, adaptive_As = [], [], []
for model in models

  uh, iter, A = pdas_signorini_solver(model,f,true)
  push!(adaptive_iters, iter[1])
  push!(adaptive_uhs, uh)
  push!(adaptive_λs, iter[3])
  push!(adaptive_As, A)
end
adaptive_iters

rm("tmp_nl", recursive=true)
if !isdir("tmp_nl")
    mkdir("tmp_nl")
end

level=7; Ω = adaptive_uhs[level][1].fe_space.fe_basis.trian
createpvd("signorini_2d") do pvd
    pvd[1] = createvtk(Ω, "tmp_nl/signorini_2d_1" * ".vtu", cellfields=["u" => adaptive_uhs[level][1]])
    for k in 2:length(adaptive_uhs[level])
      pvd[k] = createvtk(Ω, "tmp_nl/signorini_2d_$k" * ".vtu", cellfields=["u" => adaptive_uhs[level][k]])
    end
end
# writevtk(Triangulation(models[7]), "adaptive_solution.vtu", cellfields=["u"=>adaptive_uhs[end][end]])