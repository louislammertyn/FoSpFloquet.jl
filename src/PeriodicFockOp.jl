abstract type AbstractTDFockOperator end

struct PeriodicFockOperator <: AbstractTDFockOperator
    ops::Vector{MultipleFockOperator}      # store operators
    funcs::Vector{Function}                # store corresponding functions f(t)
    T::Float64
end

struct PeriodicFockMatrix{MT<:AbstractMatrix{ComplexF64}} <: AbstractTDFockOperator
    ops::Vector{MT}                        # store matrices
    funcs::Vector{Function}                # corresponding functions
    T::Float64
end

Base.size(A::PeriodicFockMatrix)  = size(first(A.ops))
Base.eltype(A::PeriodicFockMatrix) = eltype(first(A.ops))

function (A::PeriodicFockMatrix)(t::Float64)
    total = zero(first(A.ops))  # initialize to zero matrix
    for (f,o) in zip(A.funcs, A.ops)
        c = f(t)
        total += o * c
    end
    return total
end

struct FourierTerm{O}
    freqs::Vector{Int}      # Fourier indices or frequencies
    coeffs::Vector{ComplexF64}  # Fourier coefficients
    op::O                 # Operator (MultipleFockOperator or Matrix)
end

struct FourierFockOperator <: AbstractTDFockOperator
    terms::Vector{FourierTerm{MultipleFockOperator}}
    ω::Float64
end

struct FourierFockMatrix{MT<:AbstractMatrix{ComplexF64}} <: AbstractTDFockOperator
    terms::Vector{FourierTerm{MT}}
    ω::Float64
end

function (A::FourierFockMatrix)(t::Float64)
    total = zero(first(A.terms).op)  # initialize to zero matrix
    for term in A.terms
        c = sum(exp(1im * term.freqs[i] * A.ω * t) * term.coeffs[i] for i in eachindex(term.freqs))
        total += term.op * c
    end
    return total
end

Base.size(A::FourierFockMatrix)  = size(first(A.terms).op)
Base.size(A::FourierTerm)  = size(A.op)
Base.eltype(A::FourierFockMatrix) = eltype(first(A.terms).op)

function Fourier_op(O::PeriodicFockOperator; N::Int=200, tol::Float64=1e-2)
    T = O.T
    tgrid = range(0, T, length=N+1)[1:end-1]

    terms = Vector{FourierTerm{MultipleFockOperator}}()

    # Sample times to check constant functions
    t_samples = range(0, T, 10)

    for (op, f) in zip(O.ops, O.funcs)
        fvals = f.(tgrid)

        # Check if function is constant
        if all(t -> f(t) == 1, t_samples)
            push!(terms, FourierTerm([zero(Int)], [one(ComplexF64)], op))
            continue
        end

        # Fourier transform
        fourier_f = fftshift(fft(fvals)) / N
        freqs = collect(div(-N,2):div(N-1,2))

        # Threshold small coefficients
        maxf = maximum(abs.(fourier_f))
        mask = abs.(fourier_f) .> tol*maxf

        filtered_freqs = freqs[mask]
        filtered_coeffs = fourier_f[mask]

        push!(terms, FourierTerm(filtered_freqs, filtered_coeffs, op))
    end

    return FourierFockOperator(terms, 2π/T)
end

# Must add functionality to check for aliasing!!
function Fourier_op(O::PeriodicFockMatrix; N::Int=200, tol::Float64=1e-2)
    T = O.T
    tgrid = range(0, T, length=N+1)[1:end-1]

    terms = Vector{FourierTerm{ typeof(first(O.ops))}}()
    t_samples = range(0, T, 10)

    for (op, f) in zip(O.ops, O.funcs)
        fvals = f.(tgrid)

        if all(t -> f(t) == 1, t_samples)
            push!(terms, FourierTerm([zero(Int)], [one(ComplexF64)], op))
            continue
        end

        fourier_f = fftshift(fft(fvals)) / N
        freqs = collect(div(-N,2):div(N-1,2))
        maxf = maximum(abs.(fourier_f))
        mask = abs.(fourier_f) .> tol*maxf

        push!(terms, FourierTerm(freqs[mask], fourier_f[mask], op))
    end

    return FourierFockMatrix(terms, 2π/T)
end

function matrix_rep(O::PeriodicFockOperator, basis::Vector{AbstractFockState})
    M_ops = [calculate_matrix_elements(op, basis) for op in O.ops]
    return PeriodicFockMatrix(M_ops, O.funcs, O.T)
end

function matrix_rep(O::FourierFockOperator, basis::Vector{AbstractFockState})
    terms_matrix = Vector{FourierTerm{Matrix{ComplexF64}}}()

    for term in O.terms
        M = calculate_matrix_elements(term.op, basis)
        push!(terms_matrix, FourierTerm(term.freqs, term.coeffs, M))
    end

    return FourierFockMatrix(terms_matrix, O.ω)
end



