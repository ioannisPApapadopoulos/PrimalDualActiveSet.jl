using MatrixFactorizations
import LinearAlgebra: Diagonal

function project!(x, lb, ub)
    b = x;
    b[x .< lb] .= lb[x .< lb]
    b[x .> ub] .= ub[x .> ub]
    return b
end

function project(x, lb, ub)
    b = x[:];
    b[x .< lb] .= lb[x .< lb]
    b[x .> ub] .= ub[x .> ub]
    return b
end


update_residual_and_jacobian!(::Val{true}, J, r, op, vh) = Gridap.Algebra.residual!(r, op, vh)
update_residual_and_jacobian!(::Val{false}, J, r, op, vh) = Gridap.Algebra.residual_and_jacobian!(r, J, op, vh)    

function compute_inactive_set_update(::Val{1}, ::Val{true}, jac::SparseArrays.SparseMatrixCSC, cr::AbstractVector, nullsp::AbstractArray)
    fac = MatrixFactorizations.cholesky(Symmetric(jac))
    return -(fac\cr)
end
function compute_inactive_set_update(::Val{1}, ::Val{false}, jac::SparseArrays.SparseMatrixCSC, cr::AbstractVector, nullsp::AbstractArray)
    fac = MatrixFactorizations.lu(jac)
    return -(fac\cr)
end
function compute_inactive_set_update(::Val{2}, ::Val{true}, jac::SparseArrays.SparseMatrixCSC, cr::AbstractVector, nullsp::AbstractArray)
    ml = smoothed_aggregation(jac, B=nullsp[inactive,:])
    p = aspreconditioner(ml)
    up = cg(jac, -cr, Pl=p, verbose=true)
    return up
end
function compute_inactive_set_update(::Val{3}, ::Val{true}, jac::SparseArrays.SparseMatrixCSC, cr::AbstractVector, nullsp::AbstractArray)
    p = ILUZero.ilu0(jac)
    up = cg(jac, -cr, Pl=p, verbose=true)
    return up
end

## HIK
function hik(op, uh, lb::AbstractVector{T}, ub::AbstractVector{T}, linear_flag::Val, sym_pos_def::Val; 
        solver_flag::Val=Val(1),
        tol::T=1e-9, 
        max_iter::Int=1000, damping::T=one(T),
        nullsp=Vector{T}(), history::Bool=false, show_trace::Bool=true) where T

    x = copy(uh.free_values)
    n = length(x)

    iter = 0

    active_lb = Int[]
    active_ub = Int[]
    active    = Int[]
    actives   = Vector{Vector{Int}}() 

    project!(x, lb, ub)
    vh = FEFunction(uh.fe_space,x)

    r, J  = Gridap.Algebra.residual_and_jacobian(op, vh)

    for i in 1:n
        if isapprox(x[i], lb[i]) && r[i] > 0
            push!(active_lb, i)
        elseif isapprox(x[i], ub[i]) && r[i] < 0
            push!(active_ub, i)
        end
    end
    append!(active, active_lb)
    append!(active, active_ub)

    dual = zeros(T, n)
    dual[active_lb] .= r[active_lb]
    dual[active_ub] .= -r[active_ub]

    dual_lb, dual_ub = zeros(n), zeros(n)

    mask = trues(n)
    mask[active] .= false
    inactive = findall(mask)

    @. dual_lb = dual-x+lb
    @. dual_ub = dual+x-ub
    norm_residual_Ω  = norm(dual - clamp!(dual_ub, 0, Inf) - clamp!(dual_lb, 0, Inf))
    norm_residual_Ω  = (norm_residual_Ω  
        + norm(r[active_lb]-dual[active_lb])
        + norm(r[active_ub]+dual[active_ub])
        + norm(r[inactive])
    )

    show_trace && print("HIK: Iteration 0, residual norm = $norm_residual_Ω\n")
    
    if history
        us = []
        λs = []
    end

    if solver_flag==Val(2) && isempty(nullspace)
        nullsp = ones(n)
    end


    # if krylov_solver && sym_pos_def
    #     # p = CholeskyPreconditioner(J, 5)
    #     p = Preconditioners.AMGPreconditioner{SmoothedAggregation}(J)
    # end
    
    update = similar(x)
    cr_buffer = similar(x)

    while (norm_residual_Ω) > tol && (iter < max_iter)
        
        fill!(update, zero(T))
        update[active_lb] .= lb[active_lb] - x[active_lb]
        update[active_ub] .= ub[active_ub] - x[active_ub]

        @views cr = cr_buffer[1:length(inactive)]
        if ~isempty(active)
            cr = r[inactive] + J[inactive, active]*update[active]
        else
            cr = r[inactive]
        end

        jac = J[inactive,inactive]

        update[inactive] .= compute_inactive_set_update(solver_flag, sym_pos_def, jac, cr, nullsp)
        x .+= damping*update;
        vh.free_values .= x

        update_residual_and_jacobian!(linear_flag, J, r, op, vh)
        # r, J  = Gridap.Algebra.residual_and_jacobian(op, vh)
    
        # which way should the sign be?
        dual[inactive] .= zero(T);
        dual[active_lb] .= r[active_lb]
        dual[active_ub] .= -r[active_ub]
        
        active_lb = findall((dual - x + lb).>0)
        active_ub = findall((dual - ub +x) .>0)

        empty!(active)
        append!(active, active_lb)
        append!(active, active_ub)
        # active = vcat(active_lb, active_ub)
        history && push!(actives, active)

        # print(active)
        fill!(mask, true)
        mask[active] .= false
        inactive = findall(mask)

        if history
            push!(us, FEFunction(vh.fe_space,copy(vh.free_values)))
            push!(λs, copy(dual))
        end


        @. dual_lb = dual-x+lb
        @. dual_ub = dual+x-ub
        norm_residual_Ω  = norm(dual - clamp!(dual_ub, 0, Inf) - clamp!(dual_lb, 0, Inf))
        norm_residual_Ω  = (norm_residual_Ω  
            + norm(r[active_lb]-dual[active_lb])
            + norm(r[active_ub]+dual[active_ub])
            + norm(r[inactive])
        )

        iter += 1
        show_trace && print("HIK: Iteration $iter, residual norm = $norm_residual_Ω.\n")
    end
    
    if iter == max_iter
        show_trace && print("HIK: Iteration max reached")
        @warn("HIK: Iteration max reached.")
    end
    if history
        return us, [iter, actives, λs]
    else
        return vh, iter
    end
end

function ssn(op, uh, lb::AbstractVector{T}, ub::AbstractVector{T}, M::AbstractMatrix{T}; tol::T=1e-9, max_iter::Int=1000, damping=1, history=false, show_trace=true) where T
    
    x = uh.free_values[:]

    index = Vector(1:lastindex(x))
    iter = 0
    inactive = index

    active_lb = []
    active_ub = []
    active    = []
    actives   = []
    
    project!(x, lb, ub)
    vh = FEFunction(uh.fe_space,x)
    n = length(x)

    r, J  = Gridap.Algebra.residual_and_jacobian(op, vh)

    active_lb = findall(x .≈ lb) ∩ findall(r .> 0)
    active_ub = findall(x .≈ ub) ∩ findall(r .< 0)
    active = vcat(active_lb, active_ub)
    history && push!(actives, active)

    # print(active)

    dual = zeros(n)
    mr = (M \ r)
    dual[active_lb] = mr[active_lb]
    dual[active_ub] = -mr[active_ub]
    tmp_index = index[:]
    tmp_index[active] .= 0
    inactive  = findall(x->x!=0, tmp_index)
    
    norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))
    # %normResidualOmega = norm(dual - max(zeros(n,1), dual+x-ub) - max(zeros(n,1), dual-x-lb));
    # %normResidualOmega = normResidualOmega + norm(evaluatedResidual+dual);
    show_trace && print("HIK: Iteration 0, residual norm = $norm_residual_Ω\n")
    
    if history
        us = []
        λs = []
    end
    
    uD = ExtendableSparseMatrix(Diagonal(ones(n)))
    lD = ExtendableSparseMatrix(Diagonal(ones(n)))

    # jac = [J M;uD lD]

    while (norm_residual_Ω) > tol && (iter < max_iter)
        
        # update = zeros(T, n);
        # update[active_lb] = lb[active_lb] - x[active_lb]
        # update[active_ub] = ub[active_ub] - x[active_ub]


        # cr = r + [J[:, active] M[:, inactive]]*[update[active];-dual[inactive]]
        # # cr = r + J[:, active] * update[active]
   
        # full_update = [J[:,inactive] M[:,active]] \ (-cr)
        # update[inactive] = full_update[1:length(inactive)]
        # dual[active] = dual[active] + full_update[length(inactive)+1:end]
        # dual[inactive] .= zero(T);
        # x = x + damping*update;
        # vh = FEFunction(uh.fe_space,x)

        ni, na = length(inactive), length(active)
        @assert ni+na == n

        # This is now much fast via ExtendableSparse
        uDv, lDv = ones(n), ones(n)
        uDv[inactive] .= 0
        lDv[active] .= 0
        uD[:,:] = Diagonal(uDv)
        lD[:,:] = Diagonal(lDv)

        # Now much faster as long as everything is sparse (not diagonal)
        jac = sparse([J M;uD lD])

        rc = zeros(n)
        rc[active] .= (x-ub)[active]
        rc[inactive] .= dual[inactive]

        lu_jac = MatrixFactorizations.lu(jac)
        δ = lu_jac \ -[r+M*dual;rc]
        x .+= δ[1:n]
        dual .+= δ[n+1:end]
        # xx = range(0,1,100)
        # p = Plots.plot(xx, vh.(Point.(xx)))
        # display(p)
        # sleep(1)
    
        # which way should the sign be?
        
        # dual[active_lb] = r[active_lb]
        # dual[active_ub] = -r[active_ub]
        
        active_lb = findall((dual .- x .+ lb).>0)
        active_ub = findall((dual .-ub .+x) .>0)
        active = vcat(active_lb, active_ub)

        # print(active)

        tmp_index = copy(index)
        tmp_index[active] .= 0
        inactive  = findall(x->x!=0, tmp_index)
        
        # norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))
        # project!(x,lb,ub)
        vh = FEFunction(uh.fe_space,x)
        if history
            push!(us, FEFunction(vh.fe_space,copy(vh.free_values)))
            push!(λs, copy(dual))
        end

        r, J  = Gridap.Algebra.residual_and_jacobian(op, vh)
        norm_residual_Ω  = norm(dual - max.(zeros(n,1), dual+x-ub) - max.(zeros(n,1), dual-x+lb))
        norm_residual_Ω  = norm_residual_Ω  + norm(r+M*dual)
        # norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))


        iter += 1
        show_trace && print("HIK: Iteration $iter, residual norm = $norm_residual_Ω\n")
    end
    
    if iter == max_iter
        show_trace && print("HIK: Iteration max reached")
        @warn("HIK: Iteration max reached.")
    end
    if history
        return us, [iter, actives, λs]
    else
        return vh, iter
    end
end

## Benson-Munson
function reduced_residual(r::AbstractVector{T}, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T
    rr = r[:];
    rr[x .<= lb] = min.(rr[x .<= lb], zero(T))
    rr[x .>= ub] = max.(rr[x .>= ub], zero(T))
    return rr
end

function bm(op, uh, lb::AbstractVector{T}, ub::AbstractVector{T}; us::AbstractVector=[], tol::T=1e-9, max_iter::Int=1000, damping=1, history=false, show_trace=true) where T

    x = uh.free_values[:]

    index = Vector(1:lastindex(x))
    iter = 0

    project!(x, lb, ub)

    r, J  = Gridap.Algebra.residual_and_jacobian(op, uh)

    active_lb = findall(x .≈ lb) ∩ findall(r .> 0)
    active_ub = findall(x .≈ ub) ∩ findall(r .< 0)
    active = vcat(active_lb, active_ub)

    tmp_index = index[:]
    tmp_index[active] .= 0
    inactive  = findall(x->x!=0, tmp_index)

    norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))
    show_trace && print("BM: Iteration 0, residual norm = $norm_residual_Ω\n")

    n = length(x)

    if history
        us = []
    end

    while norm_residual_Ω > tol && iter < max_iter
        update = zeros(T, n)
        update[inactive] = -J[inactive, inactive] \ r[inactive]

        x = x + damping * update
        project!(x,lb,ub)
        uh = FEFunction(uh.fe_space,x)
        if history
            push!(us, uh)
        end

        r, J  = Gridap.Algebra.residual_and_jacobian(op, uh)
        norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))

        active_lb = index[x .<= lb]
        active_lb = intersect(active_lb, index[r .> zero(T)])
        active_ub = index[x .>=ub]
        active_ub = intersect(active_ub, index[r .< zero(T)])
        active = vcat(active_lb, active_ub)
        index2 = index[:]
        index2[active] .= 0
        inactive  = findall(x->x!=0, index2)

        iter += 1
        show_trace && print("BM: Iteration $iter, residual norm = $norm_residual_Ω\n")
    end
    
    if iter == max_iter
        show_trace && print("BM: Iteration max reached")
    end
    if history
        return us, iter
    else
        return uh, iter
    end
end

## SSLS

function Φ(a::AbstractVector{T}, b::AbstractVector{T}) where T
    @assert length(a) == length(b)
    a + b - sqrt.(a.^2 + b.^2)     
end

function dΦ(a::AbstractVector{T}, b::AbstractVector{T}) where T
    @assert length(a) == length(b)
    if any(abs.(a) .> 1e-6) || any(abs.(b) .> 1e-6)
        return one(T) .- a./sqrt.(a.^2 + b.^2)
    else
        return ones(T, length(a)) ./ 2
    end
end

function FB(x, r, lb, ub, wherenoconstraint, wherelbconstraint, whereubconstraint, whereequalconstraint, wherebothconstraint)
    T = eltype(x)
    out = zeros(T,length(x))
    #  FIXME add a check for all indices here
    out[wherenoconstraint] = r[wherenoconstraint]
    
    idx = whereubconstraint
    out[idx] = Φ(ub[idx] - x[idx], -r[idx])
    
    idx = wherelbconstraint
    out[idx] = Φ(x[idx]-lb[idx],r[idx])
    
    idx = wherebothconstraint
    out[idx] = Φ(x[idx]-lb[idx],Φ(ub[idx]-x[idx],-r[idx]))
    
    idx = whereequalconstraint
    out[idx] = lb[idx] - x[idx]
    return out
end

function computeScaleAndShift(x, r, lb, ub, wherenoconstraint, wherelbconstraint, whereubconstraint, whereequalconstraint, wherebothconstraint)
    T = eltype(x)
    n = length(x)
    dshift = ones(T, n)
    dscale = ones(T, n)
    
    dshift[wherenoconstraint] .= zero(T)
    dscale[wherenoconstraint] .= one(T)
    
    idx = whereubconstraint
    dshift[idx] = dΦ(ub[idx] - x[idx], -r[idx])
    dscale[idx] = dΦ(-r[idx],ub[idx]-x[idx])
    
    idx = wherelbconstraint
    dshift[idx] = dΦ(x[idx] - lb[idx], r[idx])
    dscale[idx] = dΦ(r[idx], x[idx] - lb[idx])
   
    
    idx = wherebothconstraint;
    dshift1 = dΦ(x[idx] - lb[idx], -Φ(ub[idx] - x[idx], -r[idx]))
    dscale1 = dΦ(-Φ(ub[idx]-x[idx],-r[idx]), x[idx] - lb[idx])
    dshift2 = dΦ(ub[idx]-x[idx],-r[idx])
    dscale2 = dΦ(-r[idx],ub[idx] - x[idx])
    dshift[idx] = dshift1 + dscale1.*dshift2
    dscale[idx] = dscale1.*dscale2
    
    idx = whereequalconstraint
    dshift[idx] .= one(T)
    dscale[idx] .= zero(T)
    return (dshift, dscale)
end

function ssls(op, uh, lb::AbstractVector{T}, ub::AbstractVector{T};
     us::AbstractVector=[], tol::T=1e-9, max_iter::Int=1000, damping=1) where T

    iter = 0
    x = uh.free_values[:]
    known_roots = [u.free_values for u in us]

    project!(x,lb,ub)
    
    bound_tol = 1e10
    wherenoconstraint = intersect(findall(x->x<-bound_tol, lb), findall(x->x>bound_tol, ub))
    wherelbconstraint = intersect(findall(x->x>=-bound_tol, lb), findall(x->x>bound_tol, ub))
    whereubconstraint = intersect(findall(x->x<-bound_tol,lb), findall(x->x<=bound_tol, ub))
    whereequalconstraint = findall(x->x==ub, lb)
    wherebothconstraint = intersect(findall(x->x>=-bound_tol,lb), findall(x->x<=bound_tol,ub))
    
    r, J  = Gridap.Algebra.residual_and_jacobian(op, uh)
             
    fb = FB(x, r, lb, ub, wherenoconstraint,wherelbconstraint,whereubconstraint,whereequalconstraint,wherebothconstraint)
    normFB = norm(fb)
    print("Iteration 0, residual norm = $normFB\n")
    
    while normFB > tol && iter < max_iter
        
        (dshift, dscale) = computeScaleAndShift(x, r, lb, ub, wherenoconstraint,wherelbconstraint,whereubconstraint,whereequalconstraint,wherebothconstraint)
        shiftedJacobian = Diagonal(dshift) + Diagonal(dscale) * J

        update = -shiftedJacobian \ fb
        x = x + damping*update
        uh = FEFunction(uh.fe_space,x)
        
        r, J  = Gridap.Algebra.residual_and_jacobian(op, uh)  
      
        fb = FB(x, r, lb, ub, wherenoconstraint,wherelbconstraint,whereubconstraint,whereequalconstraint,wherebothconstraint)
        normFB = norm(fb)               
        iter += 1
        print("Iteration $iter, residual norm = $normFB\n")
    end
    
    if iter == max_iter
        print("Iteration max reached")
    end
    return uh
end