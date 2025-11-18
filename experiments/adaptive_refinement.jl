using Gridap, Gridap.Geometry, Gridap.Adaptivity

ϵ = 1e-2
r(x) = ((x[1]-0.5)^2 + (x[2]-0.5)^2)^(1/2)
u_exact(x) = 1.0 / (ϵ + r(x))
order = 1
model = simplexify(CartesianDiscreteModel((0,1,0,1),(20,20)))
function adapt_fun(x)
    if x[1] ≥ 0.1
        return 1.0
    else
        return 0.0
    end
end

  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,u_exact)

  "Setup integration measures"
  Ω = Triangulation(model)
  Γ = Boundary(model)
  Λ = Skeleton(model)

  dΩ = Measure(Ω,4*order)
  dΓ = Measure(Γ,2*order)

  ηh(u)  = ∫(adapt_fun)*dΩ

  "Solve the FE problem"
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)


  "Compute error indicators"
  η = estimate(ηh,uh)


  "Mark cells for refinement using Dörfler marking
  This strategy marks cells containing a fixed fraction (0.9) of the total error"
  m = DorflerMarking(0.9999)
  I = Adaptivity.mark(m,η)

  method = Adaptivity.NVBRefinement(model)
  amodel = refine(method,model;cells_to_refine=I)
  fmodel = Adaptivity.get_model(amodel)

Ω1 = Triangulation(fmodel)

writevtk(
Ω1,"fmodel"
)

V1 = TestFESpace(fmodel,reffe)
u0 = interpolate_everywhere(x->sin(x[1]), V)# FEFunction(V, zeros(V.nfree))
u0 = interpolate(u0, V)

u1 = FEFunction(V1, zeros(V1.nfree))
u1.free_values .= u0(V1.fe_basis.trian.grid.node_coordinates)

V1d = TestFESpace(fmodel,reffe;dirichlet_tags=["boundary"])
u1d = interpolate(u1,V1d)

using Gridap
writevtk(Ω, "model", cellfields=["uh"=>u0])
writevtk(Ω1, "fmodel", cellfields=["uh"=>u1d])