# This example reproduces partial results from the paper https://doi.org/10.1103/PhysRevLett.95.260404 
# It shows how periodic driving if the Bose-Hubbard Hamiltonian can interpolate between the Mott insulating phase and the superfluid phase 

using Revise
using FoSpFloquet
using LinearAlgebra

# Define space 
L = 5
N = 5
cutoff = N

geometry = (L,)
V = U1FockSpace(geometry, cutoff, N)
lattice = Lattice(geometry)
states = all_states_U1(V)

# define the Hamiltonian
J, U = 1., 3.
H_j, H_u = Bose_Hubbard_H(V, lattice, J, U)

## define the drive 
# params

V_drive = ZeroFockOperator()
for i in 1:L 
    V_drive +=  i * ni(V, i)
end



ω = 14.
K = 2.4 * ω
T = 2π/ω
f_t(t) = K * cos(ω * t )

# convert to relevant Operators
H_t = PeriodicFockOperator([H_j+ H_u, V_drive], [triv, f_t], T)
H_fourier = Fourier_op(H_t)
H_fourier_m = matrix_rep(H_fourier, states)

# Storage
U1s = []
U2s = []
U3s = []

# Tolerances
tols = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

for tol in tols
    #U1, _ = compute_Floquet(H_fourier_m, 0., 1; tol=tol)
    #push!(U1s, U1)
    
    #U2, _ = compute_Floquet(H_fourier_m, 0., 2; tol=tol)
    #push!(U2s, U2)

    U3, _ = compute_Floquet(H_fourier_m, 0.; tol=tol*1e-4)
    push!(U3s, U3)
end

U3, _ = compute_Floquet(H_fourier_m, 0.; tol=1e-8)
norm(U3 *U3',Inf)
U3 - U2s[end]
# Check convergence by differences
println("Convergence for U1:")
for i in eachindex(U1s)
    i==1 && continue
    println("ΔU1 (tol $(tols[i])) = ", norm(U1s[i] - U1s[i-1]))
end

println("Convergence for U2:")
for i in eachindex(U2s)
    i==1 && continue
    println("ΔU2 (tol $(tols[i])) = ", norm(U2s[i] - U2s[i-1]))
end

println("Convergence for U3:")
for i in eachindex(U3s)
    i==1 && continue
    println("ΔU2 (tol $(tols[i])) = ", norm(U3s[i] - U3s[i-1]))
end

# Identity matrix
I1 = Matrix{ComplexF64}(I, size(U3s[1])...)
I2 = Matrix{ComplexF64}(I, size(U3s[1])...)

println("Convergence for U1 (unitary norm):")
for i in eachindex(U1s)
    i==1 && continue
    err_unitary = norm(U1s[i]' * U1s[i-1] - I1, Inf)
    println("tol=$(tols[i]): err_unitary = $err_unitary")
end

println("Convergence for U2 (unitary norm):")
for i in eachindex(U2s)
    i==1 && continue
    err_unitary = norm(U2s[i]' * U2s[i-1] - I2, Inf)
    println("tol=$(tols[i]): err_unitary = $err_unitary")
end

println("Convergence for U3 (unitary norm):")
for i in eachindex(U3s)
    i==1 && continue
    err_unitary = norm(U3s[i]' * U3s[i-1] - I2, Inf)
    println("tol=$(tols[i]): err_unitary = $err_unitary")
end

println("Unitarity check for U3:")
for i in eachindex(U3s)
    err_unitary = norm(U3s[i]' * U3s[i] - I2, Inf)
    println("tol=$(tols[i]): err_unitary = $err_unitary")
end