# MBPTcode

Many-body perturbation theory for molecular systems, on top of
[PySCF](https://pyscf.org/): Dyson IP/EA-ADC, MPn density matrices, coupled
cluster, GW and linear response.

Everything here is hand-written: there is no code generator, and no
generated ADC code. All physics is in plain, readable NumPy.

## Methods

**ADC** — Dyson IP/EA-ADC in the algebraic-diagrammatic-construction
hierarchy.

| branch | levels |
|---|---|
| RHF, spin-adapted (CSF basis) | ADC(2)-X, ADC(3) |
| UHF / spin-orbital | ADC(3) |

Both branches have dense and matrix-free (Davidson) routes, density
fitting, Epstein-Nesbet denominator dressing, static self-energy
corrections, and a Lanczos/continued-fraction spectral solver.

**Density matrices** — MP2 through MP5 correlated 1-RDMs, restricted and
unrestricted, dense and DF, with Laplace-fused amplitude routes. GW and
CCSD/CCSDT density matrices too.

**Coupled cluster** — CCSD and CCSDT amplitudes and lambda equations,
restricted and spin-orbital, plus EOM-CC (IP/EA/EE).

**Screening** — static RPA screened Coulomb interaction W, and the screened
C^(1) block of the screened multichannel Dyson equation.

**GW / linear response** — G0W0 and eigenvalue-self-consistent GW on the
real and imaginary axes, Casida/RPA/BSE, RPA correlation energies.

## Install

Requires Python 3.10+, NumPy, SciPy and PySCF:

```bash
pip install numpy scipy pyscf
```

There is no build step. Run from the repository root so that `src` is
importable.

## Quick start

```python
from pyscf import gto, scf
from src.SingleReference.ADC import ADCSolver

mol = gto.M(atom='O 0 0 0; H 0 0 0.958; H 0.926 0 -0.240', basis='cc-pVDZ')
mf = scf.RHF(mol).run()

e, Z = ADCSolver(mf, level='adc3').solve()
print(f"ADC(3) IP = {-e[0] * 27.2114:.3f} eV   Z = {Z[0]:.3f}")
```

See `examples/` for density fitting, Epstein-Nesbet variants, open-shell
references, several ionization states, and screened singles.

## Tests

The tests are standalone scripts that print their own verdicts:

```bash
python tests/test_adc3.py
```

## Layout

```
src/Base/               PySCF interface, constants, grids, linear algebra
src/SingleReference/
    ADC/                the ADC solvers (see ADC/__init__.py for the map)
    CC/                 CCSD/CCSDT amplitudes, lambda, EOM
    DensityMatrix/      MPn / GW / CC correlated 1-RDMs
    EpsteinNesbet/      EN denominators and shifts
    GW/                 self-energy, QP equation, imaginary axis
    LinearResponse/     Casida, RPA, BSE, Davidson
src/Solvers/            quasiparticle root finders
```

## License

MIT — see [LICENSE](LICENSE). Free for any use, including commercial and
closed-source; the only condition is that the copyright notice is kept.

A few files are third-party components under the Apache License 2.0 —
the CC DIIS routine and CCSDT amplitude equations (from
[pdaggerq](https://github.com/edeprince3/pdaggerq)) and the minimax
quadrature tables (from [GreenX](https://github.com/nomad-coe/greenX)).
They are listed in [NOTICE](NOTICE), which must be retained in
redistributions.
