"""Numeric defaults and self-energy method registry shared across src/SingleReference/."""

# Lorentzian broadening for self-energy denominators / spectral functions.
DEFAULT_BROADENING_ETA = 1e-3

# CasidaSolver-only: TDA-shortcut threshold and omega^2 clipping before sqrt().
CASIDA_NUMERICAL_EPS = 1e-6

# Chunk size for blocked exciton contractions (memory/speed tradeoff only).
DEFAULT_BLOCK_SIZE = 512

# CPHF/CPKS Z-vector solve (GWDensityMatrixSolver.solve_relaxation).
CPHF_MAX_CYCLE = 100
CPHF_TOL = 1e-9

# QP root-finding (src/Solvers/qp_equation.py).
QP_NEWTON_TOL = 1e-6
QP_NEWTON_MAX_ITER = 50
QP_BISECTION_TOL = 1e-6
QP_BISECTION_MAX_ITER = 100
QP_GRAPHICAL_TOL = 1e-8
QP_GRAPHICAL_N_OMEGA = 150
QP_GRAPHICAL_MAX_BISECTION = 100
# Smallest pole strength Z = 1/f'(w) that the 'pole_strength' root selector will
# accept as a quasiparticle; roots below it are satellites. Deep valence/semicore
# states put low-Z satellites nearer to eps_HF than the true QP root, so the
# closest-root rule picks the satellite (Ne 2s: Z=0.03 at -52.5 eV vs Z=0.89 at
# -48.1 eV). Only matters where several roots exist.
QP_Z_MIN = 0.05
# Central-difference step (Hartree) for dSigma/dw when evaluating Z.
QP_Z_DERIV_STEP = 1e-3

# spin factor
GW_DENSITY_SPIN_SUM = 4.0


# ---------------------------------------------------------------------------
# Self-energy method registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY = {
    'GW':         {'vertex_mode': 'GW',         'force_rpa_casida': True,  'needs_vertex': False, 'needs_triplet': False},
    'GW@RPA':     {'vertex_mode': 'GW',         'force_rpa_casida': True,  'needs_vertex': False, 'needs_triplet': False},
    'GW@BSE':     {'vertex_mode': 'GW',         'force_rpa_casida': False, 'needs_vertex': False, 'needs_triplet': False},
    'GW@TDHF':    {'vertex_mode': 'GW',         'force_rpa_casida': False, 'needs_vertex': False, 'needs_triplet': False},
    'GWGammaInf': {'vertex_mode': 'GWGammaInf', 'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD1':       {'vertex_mode': 'PSD1',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD2':       {'vertex_mode': 'PSD2',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
    'PSD4':       {'vertex_mode': 'PSD4',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
    'PSD5':       {'vertex_mode': 'PSD5',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD6':       {'vertex_mode': 'PSD6',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD7':       {'vertex_mode': 'PSD7',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
    'PSD8':       {'vertex_mode': 'PSD8',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD9':       {'vertex_mode': 'PSD9',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
}


def get_method_info(method):
    """
    Looks up `method` (case-insensitive) in METHOD_REGISTRY
    """
    key = method.upper()
    for registered_key, info in METHOD_REGISTRY.items():
        if registered_key.upper() == key:
            return info
    raise ValueError(
        f"Unknown self-energy method '{method}'. Available: {sorted(METHOD_REGISTRY)}"
    )
