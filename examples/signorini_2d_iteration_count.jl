using ObstacleProblemSolvers
using Gridap, LinearAlgebra

function pdas_signorini_solver(nx::Integer,ny::Integer,f::Function, history::Bool=false)
  P = SignoriniRectangle(nx,ny,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)
  return uh,iter
end

f(x) = VectorValue(0.0,10.0)

iters, uniform_iters = [], []
for nx in [5,10,20,40,80,160]#,320]
  uh, iter = pdas_signorini_solver(5*nx,5,f)
  push!(iters, iter)
  uh, iter = pdas_signorini_solver(5*nx,nx,f)
  push!(uniform_iters, iter)
end
uniform_iters
iters

nx = 100
ny = 20
@time uhs, iter_actives = pdas_signorini_solver(nx,ny,true,f);
Ω = uhs[1].fe_space.fe_basis.trian
createpvd("signorini_2d_$ny") do pvd
    pvd[1] = createvtk(Ω, "tmp_nl/signorini_2d_$(ny)_1" * ".vtu", cellfields=["u" => uhs[1]])
    for k in 2:length(uhs)
      pvd[k] = createvtk(Ω, "tmp_nl/signorini_2d_$(ny)_$k" * ".vtu", cellfields=["u" => uhs[k]])
    end
end

domain = (0,5,0,0.1)
partition = (400,40)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64}, 1)
Vu = TestFESpace(model,reffe, conformity=:H1)
ub_vtk = interpolate_everywhere(x->VectorValue(0,1+0.5), Vu)
writevtk(ub_vtk.fe_space.fe_basis.trian, "bounds_ub" * ".vtu", cellfields=["ub" => ub_vtk])  