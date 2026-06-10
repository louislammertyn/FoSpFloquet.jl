# we want the user to be able to calculate time evolution operator at arbitrary times and the floquet operator
# but also we dont want to recalculate the commutator terms so at least we need a split option between having calculated them and not
# 
function analytic_integral(t::Float64, dt::Float64, n::Int, ω::Float64)::ComplexF64
    if n == 0
        return dt
    else
        return -1im * exp(1im * n * ω * t) * (exp(1im * n * ω * dt) - 1) / ( n * ω)
    end
end

function analytic_double_integral(t::Float64, dt::Float64, n::Int, m::Int, ω::Float64)::ComplexF64
    # handle the simple special cases to avoid divide-by-zero
    if n == 0 && m == 0
        return 0.5 * dt^2
    elseif n == 0 && m != 0
        return -exp(-1im*ω*m*t)*(ω*dt*m*1im + exp(-1im*ω*dt*m) - 1)/(m^2*ω^2)
    elseif m == 0 && n != 0
        return exp(-1im*ω*n*t)*((n*ω*dt*1im + 1)*exp(-1im*n*ω*dt) - 1)/(n^2*ω^2)
    elseif n + m == 0 && n != 0 && m!=0
        # resonant case n + m = 0
        return (-n*ω*dt*1im - exp(-1im*n*ω*dt) + 1)/(n^2*ω^2)
    else
        # general case
        return exp(-1im * ω * t * (n + m)) *
               ((n + m) * exp(-1im * n * ω * dt) - n * exp(-1im * (n + m) * ω * dt) - m) /
               ((n + m) * n * m * ω^2)
    end
end

function all_commutators_Fourier(H::FourierFockMatrix)
    Commutators = Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}()
    tmp = similar(first(H.terms).op)
    tmp2 = similar(tmp)
    tmp3 = similar(tmp)

    for h1 in H.terms, h2 in H.terms 
        mul!(tmp, h1.op, h2.op) 
        mul!(tmp2, h2.op, h1.op)
        tmp .-= tmp2

        for (i,n) in enumerate(h1.freqs), (j,m) in enumerate(h2.freqs )
            !(n<m) && continue
            tmp3 .= 0
            axpy!(h1.coeffs[i] * h2.coeffs[j], tmp, tmp3)
            push!(Commutators, ((n,m), copy(tmp3)))
        end
    end
    return Commutators
end

function integrals_freq_to_index(n::Int)
    α = n>=0 ? 1 : 0
    return (2*abs(n)+α)
end
for i in -10:10 
    println(integrals_freq_to_index(i))
end
function FM_step_1st!(
    H::FourierFockMatrix, frequencies::Vector{Int}, U::Matrix{ComplexF64}, t::Float64, dt::Float64,
    cache::Tuple{Matrix{ComplexF64},Matrix{ComplexF64},Matrix{ComplexF64},Dict{Int, ComplexF64}})

    dΩ, exp_dΩ, tmp, integrals_single = cache

    # integrals
    for n in frequencies
        integrals_single[n] = analytic_integral(t,dt,n,H.ω)
    end

    fill!(dΩ, zero(ComplexF64))

    # First order sum
    for h in H.terms 
        for (i,n) in enumerate(h.freqs)
            axpy!(integrals_single[n] * h.coeffs[i], h.op, dΩ)
        end
    end

    exp_dΩ .= exp(-1im * dΩ)
    mul!(tmp, exp_dΩ, U)
    U .= tmp

end


function FM_step_2nd!(
    H::FourierFockMatrix, frequencies::Vector{Int}, comms::Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}, U::Matrix{ComplexF64}, t::Float64, dt::Float64,
    cache::Tuple{
        Matrix{ComplexF64},
        Matrix{ComplexF64},
        Matrix{ComplexF64},
        Dict{Int, ComplexF64},
        Dict{Tuple{Int, Int}, ComplexF64}
    }
)
    dΩ, exp_dΩ, tmp, integrals_single, integrals_double = cache

    # integrals
    for n in frequencies
        integrals_single[n] = analytic_integral(t,dt,n,H.ω)
        for m in frequencies
            integrals_double[(n,m)] = analytic_double_integral(t,dt,n,m,H.ω)
        end
    end


    fill!(dΩ, zero(ComplexF64))

    # First order sum
    for h in H.terms 
        for (i,n) in enumerate(h.freqs)
            axpy!(integrals_single[n] * h.coeffs[i], h.op, dΩ)
        end
    end

    # second order sum 
    for ((n,m), M) in comms
        axpy!(integrals_double[(n,m)] - integrals_double[(m,n)], M, dΩ)
    end

    #add to U
    exp_dΩ .=exp(-1im *dΩ)
    mul!(tmp, exp_dΩ, U)
    U .= tmp
end

function FM_step_combined!(
    H::FourierFockMatrix, frequencies::Vector{Int}, 
    comms::Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}},
    U1::Matrix{ComplexF64}, U2::Matrix{ComplexF64},
    t::Float64, dt::Float64,
    cache::Tuple{
        Matrix{ComplexF64}, Matrix{ComplexF64}, Matrix{ComplexF64},  # for 1st order: dΩ1, exp_dΩ1, tmp1
        Matrix{ComplexF64}, Matrix{ComplexF64}, Matrix{ComplexF64},  # for 2nd order: dΩ2, exp_dΩ2, tmp2
        Dict{Int, ComplexF64},
        Dict{Tuple{Int, Int}, ComplexF64}
    }
)
    dΩ1, exp_dΩ1, tmp1, dΩ2, exp_dΩ2, tmp2, integrals_single, integrals_double = cache

    # Compute single integrals once, shared by both orders
    for n in frequencies
        integrals_single[n] = analytic_integral(t, dt, n, H.ω)
    end

    # Compute double integrals for 2nd order
    for n in frequencies, m in frequencies
        integrals_double[(n,m)] = analytic_double_integral(t, dt, n, m, H.ω)
    end

    # Build first-order dΩ (shared first-order sum)
    fill!(dΩ1, zero(ComplexF64))
    for h in H.terms
        for (i, n) in enumerate(h.freqs)
            axpy!(integrals_single[n] * h.coeffs[i], h.op, dΩ1)
        end
    end

    # Build second-order dΩ by starting from the first-order sum
    dΩ2 .= dΩ1
    for ((n,m), M) in comms
        axpy!(integrals_double[(n,m)] - integrals_double[(m,n)], M, dΩ2)
    end

    # Apply to U1 (1st order)
    exp_dΩ1 .= exp(-1im * dΩ1)
    mul!(tmp1, exp_dΩ1, U1)
    U1 .= tmp1

    # Apply to U2 (2nd order)
    exp_dΩ2 .= exp(-1im * dΩ2)
    mul!(tmp2, exp_dΩ2, U2)
    U2 .= tmp2
end

# Fourier-Magnus time evolution (FMTE) over arbitrary interval for first order Magnus expanison per timestep
function FMTE_1st(H::FourierFockMatrix, t_i::Float64, t_e::Float64, tol::Float64=1e-2, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    # Initialize identity matrix
    U = Matrix{ComplexF64}(I, size(first(H.terms).op)...)

    # Precompute frequencies
    freqs = all_frequencies(H)

    # Cache for Magnus steps
    dΩ = similar(U); exp_dΩ = similar(U); tmp = similar(U)
    integrals_single = Dict{Int, ComplexF64}()
    
    cache = (dΩ, exp_dΩ, tmp, integrals_single)

    t = t_i
    dt = 0.01 * (2π / H.ω)  # initial timestep
    dt_min = 1e-12
    dt_max = 0.1 * (2π / H.ω)

    steps = 0
    t_start = time()

    U_temp = copy(U)
    while t < t_e

        # Compute first-order Magnus increment
        FM_step_1st!(H, freqs, U_temp, t, dt, cache)  # see below

        # Compute error = norm of Magnus increment
        err = norm(dΩ, Inf) 

        if err < tol
            # Accept step
            t += dt
            steps += 1
            U .= U_temp
            # Grow timestep if error is much smaller than tolerance
            if err < 0.2*tol
                dt *= 1.1
                dt = min(dt, dt_max)
            end
        else
            # Shrink timestep
            dt *= 0.8 * (tol / err)^(1/2)
            dt = max(dt, dt_min)
            U_temp .= U
            continue  # redo step with smaller dt
        end
    end

    println("Algorithm ran $steps steps in $(round((time()-t_start)/60,digits=2)) mins")
    return U
end



# Fourier-Magnus time evolution (FMTE) over arbitrary interval 
function FMTE_2nd(H::FourierFockMatrix, t_i::Float64, t_e::Float64, tol::Float64=1e-2, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}()) 
    U = Matrix{ComplexF64}(I, size(first(H.terms))...)  # initialize identity matrix

    # Precompute frequencies and commutators
    freqs = all_frequencies(H)
    if isempty(comms)
        comms = all_commutators_Fourier(H)
    end

    # Cache for Magnus steps
    dΩ = similar(U); exp_dΩ = similar(U); tmp = similar(U)
    ints_single = Dict(k => zero(ComplexF64) for k in freqs)
    ints_double = Dict((n,m) => zero(ComplexF64) for n in freqs, m in freqs)
    cache = (dΩ, exp_dΩ, tmp, ints_single, ints_double)

    t = t_i
    dt = 0.01 * 2*π /  H.ω  # initial timestep

    steps = 0
    t1 = time()
    U_temp = copy(U)
    while t < t_e
        
        # 2nd-order Magnus step (full and two half-steps)
        FM_step_2nd!(H, freqs, comms, U_temp, t, dt, cache)

        # Estimate error
        err = norm(dΩ, Inf) 

        if err < tol
            t += dt
            steps += 1
            U .= U_temp
            
            # Increase timestep for next iteration if error is much smaller than tol
            if err < 0.1*tol
                dt *= 1.2
                dt = min(dt, 0.1*2π/H.ω)   # max allowed
            end
        else
            # Shrink timestep
            dt *= 0.9 * (tol / err)^(1/3)
            dt = max(dt, 1e-12)  # min allowed
            U_temp .= U
        end
        
    end
    t = time() - t1
    println("algorithm ran $steps steps in $(round(t/60;digits=2)) mins")
    return U, comms
end

function FMTE(H::FourierFockMatrix, t_i::Float64, t_e::Float64, tol::Float64=1e-4, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    U = Matrix{ComplexF64}(I, size(first(H.terms).op)...)

    # Precompute frequencies and commutators
    freqs = all_frequencies(H)
    if isempty(comms)
        comms = all_commutators_Fourier(H)
    end

    # Caches for both orders
    dΩ1 = similar(U); exp_dΩ1 = similar(U); tmp1 = similar(U)
    dΩ2 = similar(U); exp_dΩ2 = similar(U); tmp2 = similar(U)
    ints_single = Dict(k => zero(ComplexF64) for k in freqs)
    ints_double = Dict((n,m) => zero(ComplexF64) for n in freqs, m in freqs)
    cache = (dΩ1, exp_dΩ1, tmp1, dΩ2, exp_dΩ2, tmp2, ints_single, ints_double)

    t      = t_i
    dt     = 0.01 * (2π / H.ω)
    dt_min = 1e-12
    dt_max = 0.1  * (2π / H.ω)

    U1_temp = copy(U)
    U2_temp = copy(U)
    steps   = 0
    t_start = time()

    while t < t_e
        dt = min(dt, t_e - t)   # don't overshoot

        # Take a step with each order (both start from the same U)
        U1_temp .= U
        U2_temp .= U
        FM_step_combined!(H, freqs,  comms, U1_temp, U2_temp, t, dt, cache)

        # Error estimate: difference between the two propagators
        err = norm(U2_temp .- U1_temp, Inf)

        if err < tol
            t     += dt
            steps += 1
            U .= U2_temp          # accept the higher-order result

            if err < 0.1 * tol
                dt = min(dt * 1.2, dt_max)
            end
        else
            # Richardson-style step-size control for a 2nd-order method
            dt = max(dt * 0.9 * (tol / err)^(1/3), dt_min)
        end
    end

    println("Algorithm ran $steps steps in $(round((time()-t_start)/60, digits=2)) mins")
    return U, comms
end

function FMTE(H::FourierFockMatrix, t_i::Float64, t_e::Float64, order::Int, tol::Float64=1e-2, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    if order ==1
        return FMTE_1st(H, t_i, t_e, tol), comms
    elseif order == 2
        return FMTE_2nd(H, t_i, t_e, tol, comms)
    else
        @error "The demanded order has not been implemented"
    end
end

mutable struct FourierMagnusEvolutionM <: AbstractMatrix{ComplexF64}
    U::Matrix{ComplexF64}
    ti::Float64
    te::Float64 
    commutators::Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}
    U_fl::Matrix{ComplexF64}
end
Base.size(A::FourierMagnusEvolutionM)  = size(A.U)
Base.getindex(A::FourierMagnusEvolutionM, i::Int, j::Int) = getindex(A.U)
Base.eltype(A::FourierMagnusEvolutionM) = eltype(A.U)

is_nontrivial(U::Matrix) = !isapprox(U, I, atol=1e-12)  # identity check

function (U2::FourierMagnusEvolutionM)∘(U1::FourierMagnusEvolutionM)
    @assert isapprox(U1.te, U2.ti; atol=1e-12)

    # Compose the evolution operators
    U_new = U2.U * U1.U

    # Inherit commutators: prefer U2 if non-empty, else U1, else empty
    comms_new = isempty(U2.commutators) ? (isempty(U1.commutators) ? Vector{Tuple{Tuple{Int,Int},Matrix{ComplexF64}}}() : U1.commutators) : U2.commutators

    # Inherit Floquet operator if nontrivial
    U_fl_new = is_nontrivial(U2.U_fl) ? U2.U_fl : U1.U_fl

    return FourierMagnusEvolutionM(U_new, U1.ti, U2.te, comms_new, U_fl_new)
end

# Convert PeriodicFockMatrix to FourierFockMatrix
function FM_U(H::PeriodicFockMatrix, t_i::Float64, t_e::Float64, tol::Float64=1e-9, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    H_fourier = Fourier_op(H)
    return FM_U(H_fourier, t_i, t_e, tol, comms)
end

# Convert PeriodicFockMatrix to FourierFockMatrix
function FM_U(H::PeriodicFockMatrix, U_fl::Matrix, t_i::Float64, t_e::Float64, tol::Float64=1e-9, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    H_fourier = Fourier_op(H)
    return FM_U(H_fourier, U_fl, t_i, t_e, tol, comms)
end

# Main function for FourierFockMatrix
function FM_U(H::FourierFockMatrix, t_i::Float64, t_e::Float64, tol::Float64=1e-9, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    @assert t_e > t_i "Make sure the times are in proper order."

    nT = div(t_e - t_i, H.T)          # number of full periods
    t_rem = rem(t_e - t_i, H.T)       # leftover micro-motion

    # case 1: multiple full periods, negligible micro-motion → use Floquet powers 
    if nT > 1 && t_rem < tol
        U_F, comms = compute_Floquet(H, t_i; tol, comms)
        return FourierMagnusEvolutionM(U_F^n, t_i, t_e, comms, U_F)
    end

    # case 2: one or more periods with leftover micro-motion → First Floquet then leftover micromotion with FMTE
    if nT >= 1
        # Compute Floquet for first period
        U_F, comms = compute_Floquet(H, t_i; tol, comms)
        # Evolve leftover micro-motion
        t_start_rem = t_i + nT * H.T
        U_rem = t_rem > tol ? FMTE(H, t_start_rem, t_e, tol, comms)[1] : Matrix{ComplexF64}(I, size(first(H.terms))...)
        return FourierMagnusEvolutionM(U_rem * U_F^nT, ti, te, comms, U_F)
    end

    # case 3: short interval (<1 period) → just FMTE
    U,comms = FMTE(H, t_i, t_e, tol, comms)
    return FourierMagnusEvolutionM(U, ti, te, comms, U_F)
end

# Compute Floquet operator in initial time gauge of choice
function compute_Floquet(H::FourierFockMatrix, t_i::Float64, order::Int; tol::Float64=1e-9, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    t_e = t_i + 2*π/H.ω
    return FMTE(H, t_i, t_e, order, tol, comms)
end

function compute_Floquet(H::FourierFockMatrix, t_i::Float64; tol::Float64=1e-9, comms=Vector{Tuple{Tuple{Int, Int}, Matrix{ComplexF64}}}())
    t_e = t_i + 2*π/H.ω
    return FMTE(H, t_i, t_e, tol, comms)
end






