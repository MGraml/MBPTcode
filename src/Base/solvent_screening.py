"""Solvent-screened Coulomb kernel v -> v + vtilde, vtilde = v chi v.

Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016),
doi:10.1063/1.4946778, embed a quantum solute (region 1) in a polarizable
continuum (region 2).  Because the two regions have no overlapping orbitals,
P_0 is block diagonal and the Dyson equation for W folds exactly onto
region 1 (their Eqs. (11)-(13)):

    W_11    = v_11^tilde + v_11^tilde P_0,11 W_11
    v^tilde = v_11 + v_12 chi_22 v_21                              (Eq. 12)
    chi_22  = P_0,22 + P_0,22 v_22 chi_22                          (Eq. 13)

chi_22 is the *reducible* (interacting) polarizability of the solvent alone.
So the whole embedding is one substitution: build every self-energy exactly as
in the gas phase, but with

    v  ->  v + vtilde ,     vtilde = v chi v ,

the second term being the reaction potential (Eq. 15) felt inside the cavity.
Nothing in the diagrammatics changes, which is why attaching a screening
object to an mf covers *every* self-energy in this code at once -- both
chokepoints that hand out two-electron integrals (dense
get_two_electron_integrals_chemist and DF get_density_fitting_coefficients)
consult it.

Optical, not static
-------------------
chi must be the solvent response at *optical* frequencies: an added electron
or hole is a fast excitation, so only the solvent's electronic degrees of
freedom follow it (eps_infinity = n^2 ~ 1.78 for water), not its nuclear
reorientation (eps ~ 78.36).  `eps` here therefore defaults to n^2 looked up
from pyscf's SMD solvent table, and a value large enough to be a static
constant is rejected unless `allow_static_eps=True`.  The paper's
non-equilibrium recipe is: SCF inside PCM(eps_static) -- pyscf's own
`solvent.PCM` -- then self-energies with v + vtilde(eps_optical), which is
what this module supplies.  See examples/09_solvated_gw_adc.py.

Discretization
--------------
vtilde is evaluated with pyscf's PCM apparent-surface-charge machinery: a
source charge distribution produces the potential v_k on cavity surface point
k, the surface responds with charges q = K^-1 R v, and the reaction potential
that those charges generate back inside the cavity is the second term of
Eq. (12).  With the symmetrized response Q = (K^-1 R + (K^-1 R)^T)/2 that
pyscf itself uses,

    vtilde(rho_1, rho_2) = sum_kk' v_k[rho_1] Q_kk' v_k'[rho_2] ,

which is negative semi-definite (screening lowers the interaction) and only
needs the one-electron grid potentials v_k, never a four-index object.
"""
import numpy as np
import scipy.linalg

from pyscf import df as pyscf_df
from pyscf import gto
from pyscf.solvent import pcm as pyscf_pcm
from pyscf.solvent.smd import solvent_db

# n^2 below this is not a plausible optical dielectric constant for a
# condensed phase; above it the caller almost certainly passed a static eps by
# mistake (water: 1.78 optical vs 78.36 static).
_MAX_PLAUSIBLE_OPTICAL_EPS = 10.0
# I + M (the screened Coulomb metric in the whitened auxiliary basis) must stay
# positive definite: v + vtilde is still a positive kernel, so an eigenvalue at
# or below this floor means the PCM discretization has over-screened.
_MIN_SCREENED_METRIC_EIGENVALUE = 1e-6


def solvent_dielectrics(name):
    """(eps_optical, eps_static) for a named solvent, from pyscf's SMD table.

    eps_optical = n^2 with n the refractive index -- the constant that screens
    a fast (photoemission/optical) excitation, and the one this module wants.
    eps_static additionally contains the nuclear reorientation of the solvent
    and belongs in the *ground-state* SCF, not here.
    """
    key = name.lower()
    if key not in solvent_db:
        raise KeyError(
            f"unknown solvent {name!r}; pyscf's SMD table has "
            f"{len(solvent_db)} entries, e.g. 'water', 'acetonitrile', "
            f"'toluene' -- or pass eps= explicitly")
    descriptors = solvent_db[key]
    return descriptors[0] ** 2, descriptors[5]


def resolve_optical_eps(eps=None, solvent=None, allow_static_eps=False):
    """(eps_optical, eps_static_or_None) from either an explicit eps or a
    solvent name, with the optical-vs-static guard.

    Split out from SolventScreening so every caller rejects a static constant
    the same way.
    """
    if (eps is None) == (solvent is None):
        raise ValueError("pass exactly one of eps= or solvent=")
    if solvent is not None:
        eps, eps_static = solvent_dielectrics(solvent)
    else:
        eps_static = None
    if eps < 1.0:
        raise ValueError(f"eps = {eps} < 1 is not a dielectric constant")
    if eps > _MAX_PLAUSIBLE_OPTICAL_EPS and not allow_static_eps:
        raise ValueError(
            f"eps = {eps} looks like a *static* dielectric constant. The "
            f"screening of an added electron or hole is optical: use "
            f"eps = n^2 (water 1.78, not 78.36) -- solvent_dielectrics() "
            f"returns both. Pass allow_static_eps=True to force it (that "
            f"is the equilibrium limit, correct only for a process slow "
            f"enough for the solvent nuclei to relax).")
    return eps, eps_static


class SolventScreening:
    """The reaction-field kernel vtilde = v chi v of a PCM continuum.

    Hands out vtilde in whichever representation a consumer needs:
      * `kernel_mo`      -- dense (pq|rs) chemist-order correction, for the
                            four-index route;
      * `whitened_transform` -- the (naux, naux) matrix T with
                            B -> T B turning a Coulomb-metric RI factor into
                            one that reproduces (pq|v + vtilde|rs), for the DF
                            route.

    Both come from the same surface response, so the two routes agree to the
    RI error of the underlying B (tests/test_solvent_screening.py pins this).
    """

    def __init__(self, mol, eps=None, solvent=None, method='IEF-PCM',
                 lebedev_order=29, vdw_scale=1.2, r_probe=0.0,
                 radii_table=None, allow_static_eps=False, static_cohsex=True):
        """eps: optical dielectric constant. Give this or `solvent` (a name in
        pyscf's SMD table, whose refractive index sets eps = n^2), not both.
        method/lebedev_order/vdw_scale/r_probe/radii_table are handed straight
        to pyscf's PCM and carry its meanings and defaults.

        static_cohsex: also apply the first-order reaction-field operator
        `cohsex_correction` wherever a consumer builds a static self-energy.
        On by default -- see that method for why the v -> v + vtilde
        substitution alone loses the polarization energy. Set False only to
        study the substitution in isolation.
        """
        if hasattr(mol, 'lattice_vectors'):
            raise NotImplementedError(
                "PCM screening is molecular only -- a periodic Cell has no "
                "cavity to carve.")
        eps, eps_static = resolve_optical_eps(eps, solvent, allow_static_eps)

        self.mol = mol
        self.eps = eps
        self.eps_static = eps_static
        self.solvent = solvent
        self.method = method
        self.static_cohsex = static_cohsex

        self._pcm = pyscf_pcm.PCM(mol)
        self._pcm.eps = eps
        self._pcm.method = method
        self._pcm.lebedev_order = lebedev_order
        self._pcm.vdw_scale = vdw_scale
        self._pcm.r_probe = r_probe
        self._pcm.radii_table = radii_table
        self._pcm.verbose = 0
        self._pcm.build()

        self._response = None
        self._v_ao = None
        self._aux_cache = {}
        self._transform_cache = {}

    # ---- surface response -------------------------------------------------

    @property
    def ngrids(self):
        return self._pcm.surface['grid_coords'].shape[0]

    def response_matrix(self):
        """Symmetrized apparent-surface-charge response Q: q = Q v.

        pyscf solves q = K^-1 R v and symmetrizes as (q + R^T K^-T v)/2; the
        same symmetrization here makes vtilde exactly symmetric under
        (pq) <-> (rs), which the four-index and DF consumers both require.
        Negative semi-definite: a positive test charge induces negative
        surface charge, so the reaction potential screens.
        """
        if self._response is None:
            K = self._pcm._intermediates['K']
            R = self._pcm._intermediates['R']
            KiR = np.linalg.solve(K, R)
            self._response = 0.5 * (KiR + KiR.T)
        return self._response

    def _fakemol(self):
        """Gaussian-smeared surface point charges -- the same regularized
        charges pyscf's PCM uses in _get_v/_get_vmat, so our grid potentials
        and its K/R matrices refer to one and the same discretization."""
        surface = self._pcm.surface
        return gto.fakemol_for_charges(surface['grid_coords'],
                                       expnt=surface['charge_exp'] ** 2)

    # ---- grid potentials of the source distributions ----------------------

    def ao_grid_potential(self, mol):
        """v_k[phi_mu phi_nu], shape (nao, nao, ngrids). Cached: this is the
        one genuinely expensive integral in the module."""
        if self._v_ao is None:
            self._check_mol(mol)
            self._v_ao = pyscf_df.incore.aux_e2(
                mol, self._fakemol(), intor='int3c2e', aosym='s1')
        return self._v_ao

    def mo_grid_potential(self, mol, mo_coeff):
        """v_k[phi_p phi_q], shape (nmo, nmo, ngrids)."""
        v_ao = self.ao_grid_potential(mol)
        half = np.tensordot(mo_coeff, v_ao, axes=(0, 0))        # (p, nu, k)
        return np.tensordot(mo_coeff, half, axes=(0, 1)).transpose(1, 0, 2)

    def aux_grid_potential(self, auxmol):
        """v_k[chi_P] for auxiliary basis functions, shape (naux, ngrids) --
        the paper's ingredient for its Eq. (16) vtilde_{beta beta'}."""
        return gto.mole.intor_cross('int2c2e', auxmol, self._fakemol())

    # ---- vtilde in the two representations consumers need -----------------

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        """vtilde as a dense chemist-order (pq|rs) correction.

        Rows (pq) are built from `mo_bra`, columns (rs) from `mo_ket` (default:
        the same), so the UHF aa / ab / bb blocks each get their own call.
        """
        v_bra = self.mo_grid_potential(mol, mo_bra)
        v_ket = (v_bra if mo_ket is None or mo_ket is mo_bra
                 else self.mo_grid_potential(mol, mo_ket))
        n_bra, n_ket, ng = v_bra.shape[0], v_ket.shape[0], self.ngrids
        screened = v_bra.reshape(-1, ng) @ self.response_matrix()
        return (screened @ v_ket.reshape(-1, ng).T).reshape(
            n_bra, n_bra, n_ket, n_ket)

    def kernel_ao(self, mol):
        """vtilde as a dense chemist-order (mu nu|lambda sigma) correction."""
        v_ao = self.ao_grid_potential(mol)
        nao, ng = v_ao.shape[0], self.ngrids
        screened = v_ao.reshape(-1, ng) @ self.response_matrix()
        return (screened @ v_ao.reshape(-1, ng).T).reshape(nao, nao, nao, nao)

    def cohsex_correction(self, mol, mo_coeff, nocc):
        """Static COHSEX self-energy of the reaction field, (nmo, nmo) in the MO
        basis of `mo_coeff` -- the *first order in vtilde* term that the
        v -> v + vtilde substitution cannot generate by itself.

        Why it is needed.  Every correlated method here is built on a mean
        field whose exchange operator is the BARE Sigma_x = -sum_occ v: the ADC
        secular matrix starts at Sigma^(2), and the GW quasiparticle equation
        adds only Sigma_c on top of eps_HF.  Substituting v -> v + vtilde
        therefore reaches those methods only at order vtilde*v and beyond --
        the leading reaction-field term drops out, and with it essentially all
        of the polarization energy.  What is missing is exactly the paper's
        static COHSEX operator (its Eqs. (20)-(22)) evaluated with vtilde in
        place of (W - v):

            Sigma^SEX_pq = - sum_i^occ  (p i|vtilde|i q)
            Sigma^COH_pq = 1/2 sum_n^all (p n|vtilde|n q)     [sum_n |n><n| ~ delta]

        so that, writing the two together,

            Sigma^solv = 1/2 ( sum_a^virt - sum_i^occ ) (p .|vtilde|. q) .

        The sign structure is the classical image-charge result: with vtilde
        locally constant at -lambda the diagonal is +lambda/2 for an occupied
        level and -lambda/2 for a virtual one, i.e. the Born stabilization
        -q^2/2a (1 - 1/eps) of both the cation and the anion.  The IP drops and
        the EA rises by the same amount; the gap closes.

        Add it to the static (one-body, frequency-independent) part of whatever
        self-energy is being solved -- Sigma(infinity) for ADC,
        `xc_correction` for the GW quasiparticle equation.  It is first order
        in vtilde and zeroth order in v, so it does not double count anything
        the substituted interaction produces (those terms all carry at least
        one bare v).

        nocc: occupied orbital count *in this spin channel* (for a closed-shell
        restricted reference, the doubly-occupied count -- exchange is
        same-spin, so there is no factor of two).
        """
        v_mo = self.mo_grid_potential(mol, mo_coeff)
        nmo = v_mo.shape[0]
        if not 0 <= nocc <= nmo:
            raise ValueError(f"nocc={nocc} outside [0, {nmo}]")
        responded = np.tensordot(v_mo, self.response_matrix(), axes=(2, 1))
        virt = np.einsum('pak,aqk->pq', responded[:, nocc:], v_mo[nocc:],
                         optimize=True)
        occ = np.einsum('pik,iqk->pq', responded[:, :nocc], v_mo[:nocc],
                        optimize=True)
        return 0.5 * (virt - occ)

    def aux_kernel(self, auxmol):
        """vtilde_{PQ} between auxiliary functions -- the paper's Eq. (16)."""
        key = _auxmol_key(auxmol)
        if key not in self._aux_cache:
            v_aux = self.aux_grid_potential(auxmol)
            self._aux_cache[key] = v_aux @ self.response_matrix() @ v_aux.T
        return self._aux_cache[key]

    def whitened_transform(self, mol, mf):
        """T with (B -> T B) turning a Coulomb-fitted RI factor into a
        (v + vtilde)-fitted one.

        With E_P,pq = (P|pq), J_PQ = (P|Q) and the RI-V ansatz that every pair
        density is replaced by its Coulomb-metric fit c = J^-1 E, substituting
        v -> v + vtilde changes only the auxiliary-basis metric:

            (pq|v + vtilde|rs) = c_pq^T (J + vtilde_aux) c_rs .

        pyscf's cderi is B = L^-1 E with L the lower Cholesky factor of J, so
        c = L^-T B and the whole substitution collapses to a single naux x naux
        congruence of the *whitened* Coulomb metric (which is the identity in
        the gas phase -- hence T = I when vtilde = 0):

            (pq|v + vtilde|rs) = B_pq^T (I + M) B_rs ,  M = L^-1 vtilde_aux L^-T
            T = (I + M)^(1/2) .

        Every DF consumer downstream keeps working unchanged: it still sees a
        plain three-index factor whose square is the interaction.
        """
        auxmol = auxmol_of(mf, mol)
        key = _auxmol_key(auxmol)
        if key in self._transform_cache:
            return self._transform_cache[key]

        j2c = auxmol.intor('int2c2e')
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
        except scipy.linalg.LinAlgError as err:
            raise RuntimeError(
                f"the auxiliary Coulomb metric of {auxmol.basis} is not "
                f"positive definite, so pyscf's cderi did not come from a "
                f"Cholesky factorization and the whitened transform below "
                f"would be built against the wrong factor. Use a "
                f"better-conditioned auxiliary basis, or run without DF.") from err
        _verify_cderi_is_cholesky(mol, mf, auxmol, low)

        v_tilde = self.aux_kernel(auxmol)
        M = scipy.linalg.solve_triangular(low, v_tilde, lower=True)
        M = scipy.linalg.solve_triangular(low, M.T, lower=True).T
        M = 0.5 * (M + M.T)

        w, U = np.linalg.eigh(np.eye(M.shape[0]) + M)
        if w.min() < _MIN_SCREENED_METRIC_EIGENVALUE:
            raise RuntimeError(
                f"the screened Coulomb metric I + M has a smallest eigenvalue "
                f"of {w.min():.3e}, i.e. the PCM reaction field over-screens "
                f"the bare interaction to the point of making it indefinite. "
                f"v + vtilde must stay a positive kernel. Check eps={self.eps} "
                f"and the cavity (lebedev_order/vdw_scale).")
        transform = (U * np.sqrt(w)) @ U.T
        self._transform_cache[key] = transform
        return transform

    # ---- housekeeping -----------------------------------------------------

    def _check_mol(self, mol):
        if mol.nao != self.mol.nao or mol.natm != self.mol.natm:
            raise ValueError(
                f"screening was built for a molecule with {self.mol.natm} "
                f"atoms / {self.mol.nao} AOs but is being asked for integrals "
                f"over one with {mol.natm} / {mol.nao}. Re-attach the "
                f"screening to the mean field you are actually running.")

    def __repr__(self):
        label = f"solvent={self.solvent!r}, " if self.solvent else ""
        return (f"SolventScreening({label}eps={self.eps:.4f} (optical), "
                f"method={self.method!r}, ngrids={self.ngrids})")


def auxmol_of(mf, mol=None):
    """The auxiliary Mole behind mf.with_df (built on demand if pyscf has not
    materialized it yet)."""
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError("mf carries no with_df -- there is no auxiliary basis")
    if getattr(with_df, 'auxmol', None) is not None:
        return with_df.auxmol
    return pyscf_df.addons.make_auxmol(mol if mol is not None else mf.mol,
                                       with_df.auxbasis)


def _auxmol_key(auxmol):
    return (auxmol.nbas, auxmol.nao,
            auxmol._bas.tobytes(), auxmol._env.tobytes())


def _verify_cderi_is_cholesky(mol, mf, auxmol, low):
    """Assert mf's cderi really is L^-1 (P|mu nu) for the L we just built.

    whitened_transform's algebra is exact for that convention and *silently
    wrong* for any other choice of fitting factor X with X^T X = J^-1 (they
    differ by an orthogonal rotation of the auxiliary index, which leaves the
    gas-phase ERIs invariant and so cannot be caught by any ERI check). One
    AO pair is enough to pin the rotation down, so this costs a single
    int3c2e shell block.
    """
    cderi_head = next(iter(mf.with_df.loop(blksize=auxmol.nao)))
    naux = auxmol.nao
    if cderi_head.shape[0] > naux:
        raise RuntimeError(
            f"cderi has more auxiliary vectors ({cderi_head.shape[0]}) than "
            f"the auxiliary basis has functions ({naux})")
    got = np.zeros(naux)
    offset = 0
    for block in mf.with_df.loop(blksize=naux):
        # _cderi is packed s2ij over AO pairs; column 0 is the (0,0) pair.
        got[offset:offset + block.shape[0]] = block[:, 0]
        offset += block.shape[0]
    if offset != naux:
        raise RuntimeError(
            f"cderi has {offset} auxiliary vectors but the auxiliary basis has "
            f"{naux}: pyscf dropped linearly dependent auxiliary functions, so "
            f"its fitting factor is not the Cholesky factor of int2c2e that "
            f"whitened_transform inverts against. Use a better-conditioned "
            f"auxiliary basis, or run the dense (df=False) route.")

    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                                 shls_slice=(0, 1, 0, 1, 0, auxmol.nbas))
    expected = scipy.linalg.solve_triangular(low, e3c[0, 0], lower=True)
    scale = max(np.abs(expected).max(), 1.0)
    if not np.allclose(got, expected, rtol=0, atol=1e-8 * scale):
        raise RuntimeError(
            f"mf.with_df's fitting factor is not L^-1 (P|mu nu) for the "
            f"Cholesky L of int2c2e (max deviation "
            f"{np.abs(got - expected).max():.3e}). whitened_transform's "
            f"algebra assumes pyscf's default DF convention; this mean field "
            f"uses a different one, so screening it through the DF route "
            f"would be silently wrong. Run the dense (df=False) route.")


# ---- attachment: one object on mf, consulted by both integral chokepoints --

def attach_solvent_screening(mf, eps=None, solvent=None, method='IEF-PCM',
                             mol=None, **kwargs):
    """Make every post-SCF two-electron integral this code hands out use
    v + vtilde instead of v, and return `mf`.

    This is a *post-SCF* substitution: it does not touch mf's orbitals, orbital
    energies or Fock matrix. Ground-state polarization belongs in the SCF via
    pyscf's own `solvent.PCM(mf)` at the static eps; this adds the optical
    response of the solvent to the correlated part, which is exactly the
    non-equilibrium split of the paper (its Section II D).

    Keyword arguments go to SolventScreening; `static_cohsex=False` there
    turns off the first-order reaction-field operator that consumers add to
    their static self-energy (see SolventScreening.cohsex_correction).
    """
    screening = SolventScreening(mol if mol is not None else mf.mol,
                                 eps=eps, solvent=solvent, method=method,
                                 **kwargs)
    mf.with_screening = screening
    keys = getattr(mf, '_keys', None)
    if keys is not None:
        keys.add('with_screening')      # keep pyscf's check_sanity quiet
    return mf


def detach_solvent_screening(mf):
    """Drop the screening, returning mf to gas-phase integrals."""
    mf.with_screening = None
    return mf


def solvent_static_selfenergy(mf, mol=None):
    """Static COHSEX reaction-field self-energy of the screening attached to
    `mf`, in that mean field's own spatial MO basis, or None.

    RHF: one (nmo, nmo) array. UHF: an (alpha, beta) pair, each built from its
    own MO coefficients and occupation. Returns None when no screening is
    attached, or when it was created with static_cohsex=False -- so callers can
    add it unconditionally, the way build_ks_static_correction is added.
    """
    screening = get_solvent_screening(mf)
    if screening is None or not screening.static_cohsex:
        return None
    mol = mol if mol is not None else mf.mol
    mo_coeff = mf.mo_coeff
    if isinstance(mo_coeff, (tuple, list)) or np.asarray(mo_coeff).ndim == 3:
        mo_a, mo_b = mo_coeff
        nocc_a, nocc_b = mf.nelec
        return (screening.cohsex_correction(mol, mo_a, nocc_a),
                screening.cohsex_correction(mol, mo_b, nocc_b))
    return screening.cohsex_correction(mol, mo_coeff, mol.nelectron // 2)


def get_solvent_screening(mf):
    """The SolventScreening attached to mf, or None. The hook every integral
    chokepoint calls -- gas-phase callers pay one getattr."""
    return getattr(mf, 'with_screening', None)
