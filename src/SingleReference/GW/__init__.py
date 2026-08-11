"""GW and vertex-corrected (PSD) self-energies and quasiparticle energies.

transition_amplitudes.py (GW chi/vertex transition amplitudes -- renamed from
amplitudes.py to avoid clashing with CC/amplitudes.py, the generated
CCSDT equations), self_energy.py, qp_energy.py, imaginary_axis.py, and
gw_polarizability.py (G0W@CC, uses CC/eom.py).

Davidson note: the three iterative eigensolvers
in this tree -- LinearResponse/davidson.py (symplectic Casida via pyscf
real_eig), ADC (root-following pyscf davidson1), CC/eom.py
(non-Hermitian pyscf davidson_nosym1 + biorthogonalization) -- all delegate
the iterative core to pyscf and differ only in problem-specific setup, so
there is no shared Davidson core worth extracting.
"""
