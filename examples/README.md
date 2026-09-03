# Examples

Each file is self-contained and runnable: `python examples/01_adc3.py`.
All use H2O/cc-pVDZ unless noted, so the numbers are directly comparable.

| file | what it shows |
|---|---|
| `01_adc3.py` | standard ADC(3), spin-free, matrix-free — the default route |
| `02_adc3_df.py` | the same with density fitting, for large bases |
| `03_en_adc3.py` | EN-ADC(3), spin-adapted — the production EN variant |
| `06_open_shell.py` | UHF / spin-orbital ADC(3) |
| `07_several_states.py` | several ionization states with pole strengths |
| `08_en_adc3_screened_singles.py` | EN-ADC(3) singles shift, bare vs RPA-screened (BSE-kernel channel split) |
| `09_isdf_gw_space_time.py` | cubic-scaling GW: an ISDF SCF, then the space-time self-energy on its factors |

## Choosing a route

**Closed shell** → `ADCSolverRestricted` (spin-free). Add `B_aa=` for DF once the
dense ERI stops fitting. **Open shell** → `ADCSolver` (spin-orbital).

**EN** is ADC(3) only. Pass the same `u2_denom_dress` dict to both the solver and
the static correction — dressing one and not the other silently mixes methods.

- `spin_adapted=True` + `shift='sum'` (2J−K): production. `'mean'` (J−K/2) and
  `'opposite'` (J) are the other weightings.
- `singles=False` dresses the doubles only — the calibrated variant.
- `singles='screened'` replaces the bare CIS/BSE diagonal J−K by the
  RPA-screened J_W−K (restricted only): see `08_en_adc3_screened_singles.py`.

`ADCSolver.u2_denom_dress` raises: its U^(2) blocks are a merged form that is
only valid for the bare amplitude.

## Expected output

```
01  IP = 12.225 eV   Z = 0.936
02  IP = 12.226 eV   Z = 0.936   (DF)
03  IP = 11.875 eV   Z = 0.929   (EN, spin-adapted)
06  IP =  8.965 eV   Z = 0.953   (UHF / spin-orbital)
09  E(ISDF-SCF) = -76.02774739 Ha   G0W0 HOMO = -12.155 eV   (space-time)
```

## Large systems

`09_isdf_gw_space_time.py` is the production route once the dense `cderi` stops
fitting: the SCF and the GW share ONE separable (ISDF) factorization, which is
both the cheaper choice and the correct one — pairing factors from one fit with
a screened interaction from another stays self-consistent and silently moves the
spectrum. Pass `freq_block=` or `scratch_dir=` to `solve_qp_energy_space_time`
when even the frequency-axis W does not fit; the answer is unchanged.
