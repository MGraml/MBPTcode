"""Standalone single-reference MBPT backend: quasiparticle energies, excitation energies, RPA correlation energies, and 1-RDMs for a HF/KS reference.

Density matrices from here (compute_gw_density_matrix) and from
src/SingleReference/CC (compute_ccsd/ccsdt_density_matrix) are both AO-basis
1-RDMs, usable interchangeably as calc_qp_energy(..., dm_correction=...).
"""
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.LinearResponse.casida import CasidaSolver, CasidaResult
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.DensityMatrix.density_matrix import (GWDensityMatrixSolver,
                                                 compute_gw_density_matrix,
                                                 MP2DensityMatrixSolver,
                                                 compute_mp2_density_matrix,
                                                 compute_mp2_density_matrix_ao,
                                                 MP3DensityMatrixSolver,
                                                 compute_mp3_density_matrix)
from src.SingleReference.GW.imaginary_axis import (
    solve_screening_imaginary_axis, self_energy_imaginary_axis,
    self_energy_matrix_imaginary_axis, solve_qp_energy_imaginary_axis)
from src.SingleReference.LinearResponse.rpa_energy import (
    rpa_correlation_energy_casida, rpa_correlation_energy_imaginary_axis)
from src.SingleReference.ADC import (
    ADCSolver, ADCSolverRestricted, ADCSolverUnrestricted, build_static_correction)
