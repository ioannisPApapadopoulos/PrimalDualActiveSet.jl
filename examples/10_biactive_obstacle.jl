using PrimalDualActiveSet, Gridap

# Biactive Obstacle Problem #1
φ(x) = 0.0
function f(x)
    if x[1] <= 0
        return 0.0
    else
        return 12*x[1]^2
    end
end
function g(x)
    if x[1] <= 0
        return 0.0
    else
        return -x[1]^4
    end
end

iters_1 = Int[]
for n in 2 .^(4:10)
    P = BiactiveObstacle(n,f,φ,g)
    u0 = interpolate_everywhere(x->0.0, P.U)
    vh, iter = hik(P, u0, max_iter=1000, history=false)
    push!(iters_1, iter[1])
end


# Biactive Obstacle Problem #2
φ(x) = 0.0
function g(x)
    if x[1]^2 + x[2]^2 < 0.25
        return -(1-4*x[1]^2-4*x[2]^2)^4
    else
        return 0.0
    end
end

function d2g(x)
    if x[1]^2 + x[2]^2 < 0.25
        return -(32*(-4*x[1]^2-4*x[2]^2+1)^2*(28*x[1]^2+4*x[2]^2-1) + 32*(-4*x[2]^2-4*x[1]^2+1)^2*(28*x[2]^2+4*x[1]^2-1))
    else
        return 0.0
    end
end
function f(x)
    if x[1]^2 + x[2]^2 > 0.75
        return -d2g(x) + 1.0
    else
        return -d2g(x)
    end
end


iters_2 = Int[]
for n in 2 .^(4:10)
    P = BiactiveObstacle(n,f,φ,g)
    u0 = interpolate_everywhere(x->0.0, P.U)
    vh, iter = hik(P, u0, max_iter=1000, history=false)
    push!(iters_2, iter[1])
end
