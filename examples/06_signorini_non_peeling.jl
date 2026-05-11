using PrimalDualActiveSet, Gridap, LinearAlgebra
using Plots, LaTeXStrings


# 2D Problem 3 Peeling
function pdas_signorini_solver(nx::Integer,ny::Integer,f::Function, history::Bool=false)
  P = SignoriniRectangle(nx,ny,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)

  A = Gridap.Algebra.jacobian(P.op, zeros(num_free_dofs(P.V)))
  return uh,iter,A
end
f(x) = VectorValue(0.0,10.0)
nx, ny = 100, 20
uhs, iter_actives = pdas_signorini_solver(nx,ny,f,true);
Ω = uhs[1].fe_space.fe_basis.trian
mkdir("tmp_nl")
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

# 3D Problem 3 Peeling
function pdas_signorini_solve_3d(nx::Integer,ny::Integer,nz::Integer,f::Function, history::Bool=false)
  P = SignoriniBox(nx,ny,nz,f)
  u0 = FEFunction(P.V, zeros(P.V.nfree))
  uh, iter = hik(P, u0, max_iter=150, history=history)
  return uh,iter
end
f(x) = VectorValue(0.0,0.0,10.0)

nx,ny,nz = 100,20,20
uhs, iter_actives = pdas_signorini_solver_3d(nx,ny,nz,f,true);
Ω = uhs[1].fe_space.fe_basis.trian
createpvd("signorini_3d_$ny") do pvd
    pvd[1] = createvtk(Ω, "tmp_nl/signorini_3d_$(ny)_1" * ".vtu", cellfields=["u" => uhs[1]])
    for k in 2:length(uhs)
      pvd[k] = createvtk(Ω, "tmp_nl/signorini_3d_$(ny)_$k" * ".vtu", cellfields=["u" => uhs[k]])
    end
end