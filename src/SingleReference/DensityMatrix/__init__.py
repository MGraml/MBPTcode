"""Correlated 1-RDMs (MPn, GW) for use as static-correction inputs in GW and ADC.

The GENERATED pipelines are canonical
and power ALL MPn production entry points:

  mpn_density_driver_restricted.py  restricted (spin-blocked) generated
   + generated_mpn_restricted/      pipeline through MP4, with generated
                                    Laplace fusion for MP4. CANONICAL RHF
                                    engine: powers compute_mp2/mp3_density_
                                    matrix_ao and ADC's _mp2/_mp3_dgamma_
                                    spatial (validated identical to the
                                    hand-written path to ~1e-16 incl. ncore
                                    and CPHF relax; MP3 runs laplace_ntau=6
                                    and never builds T4^(2) -- the generator
                                    emits t4-free m3 bodies since rank 4
                                    cannot couple to <D2| via a one-body op).
  mpn_density_driver_unrestricted.py unrestricted (separate alpha/beta)
   + generated_mpn_unrestricted/    generated pipeline, MP2+MP3. CANONICAL
                                    UHF engine: powers the UHF branches of
                                    compute_mp2/mp3_density_matrix_ao and
                                    build_mp2/mp3_static_correction
                                    (validated vs the hand-written solvers
                                    to ~1e-18 on all six per-spin blocks,
                                    closed- and open-shell:
                                    tests/test_mpn_density_unrestricted.py).
  mpn_density_driver.py             generated spin-orbital recursion through
   + generated_mpn/                 MP4. Slow; the order-agnostic oracle.

Hand-written code in density_matrix.py remains ONLY as test oracles:
  - the interleaved spin-orbital MP2/MP3DensityMatrixSolver classes are
    FCI-validated anchors (test_mp3_fci_lambda);
  - MP2/MP3DensityMatrixSolverUnrestricted are the cross-check oracles for
    both generated drivers (test_mpn_density_restricted/unrestricted.py).
    NOTE their gamma3 ncore>0 path had a genuine frozen-core bug (active
    loop indices used as absolute indices into full-size g in the t3 slice
    helpers), found via the generated driver and fixed by
    pre-windowed delegation -- see compute_gamma3_blocks's docstring.
Do not route new production code through the hand-written solvers.
"""
