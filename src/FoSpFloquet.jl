module FoSpFloquet

using Reexport
#using QuantumFockCore
@reexport using FoSpDynamics
using LinearAlgebra
using FFTW


include("./PeriodicFockOp.jl")
include("./utils.jl")
include("./FourierMagnusTE.jl")


export AbstractTDFockOperator,
       PeriodicFockOperator,
       PeriodicFockMatrix,
       FourierTerm,
       FourierFockOperator,
       FourierFockMatrix,
       Fourier_op,
       matrix_rep
       
export all_frequencies, triv


export analytic_integral,
       analytic_double_integral,
       all_commutators_Fourier,
       FM_step_1st!, FM_step_2nd!, FM_step_combined!,
       FM_U,
       compute_Floquet,
       FMTE, FMTE_1st, FMTE_2nd

export PieceWiseOperator, PieceWiseMatrix
end
