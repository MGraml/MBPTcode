import numpy as np


def get_occ_virt_indices(eps, nocc):
    """Split an orbital-energy array into occ = [0, nocc), virt = [nocc, norb) index arrays."""
    norb = len(eps)
    return np.arange(nocc), np.arange(nocc, norb)
