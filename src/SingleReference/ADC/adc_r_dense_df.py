"""Restricted dense supermatrix with every integral slice gathered from
the DF factor B_aa (never materializing the norb^4 tensor)."""
from src.SingleReference.ADC.adc_r_utils import _build_g_blocks_df
from src.SingleReference.ADC.adc_r_dense_full import _assemble


def build_supermatrix(s, nocc, static_correction=None):
    """(nH, nH) restricted supermatrix at s.level, DF-integral route."""
    blk = _build_g_blocks_df(s.B_aa, nocc, s.norb - nocc, s.norb)
    return _assemble(s, nocc, static_correction, blk)
