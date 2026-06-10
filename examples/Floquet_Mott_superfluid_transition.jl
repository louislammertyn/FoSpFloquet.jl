# This example reproduces partial results from the paper https://doi.org/10.1103/PhysRevLett.95.260404 
# It shows how periodic driving if the Bose-Hubbard Hamiltonian can interpolate between the Mott insulating phase and the superfluid phase 

using Revise
using FoSpFloquet
using LinearAlgebra
using Plots

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


amount = 20
Ks = LinRange(0, 3*ω, amount)
Ks
ϵs = zeros(Float64, length(states), amount)
for (i,k) in enumerate(Ks)
    f_t(t) = k * cos(ω * t)
    H_t = PeriodicFockOperator([H_j+ H_u, V_drive], [triv, f_t], T)
    H_fourier = Fourier_op(H_t)
    H_fourier_m = matrix_rep(H_fourier, states)

    U_fl,_ = compute_Floquet(H_fourier_m, 0.; tol=1e-5)
    ϵ, vs = eigen(U_fl)
    ϵs[:,i] = sort(real.(-1im .* log.(ϵ) ./ T))
end

pl_es = plot(;legend=false);
for i in 1:length(states)
    scatter!(pl_es, Ks / ω, ϵs[i, :], color=:red, markersize=1)
end
display(pl_es)


