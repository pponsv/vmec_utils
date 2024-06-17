import numpy as np


def make_coef_array(
    coefs, xm, xn, len_th, len_ph, deriv_order=0, deriv_dir=""
):
    """Make a 3D array of coefficients with the correct shape to be inverted."""
    assert deriv_dir in [
        "th",
        "ph",
        "",
    ], "Invalid derivative dir"  # "" means no derivative
    coef_array = np.zeros(
        (coefs.shape[0], len_th, len_ph), dtype=np.complex128
    )
    xm_mod = np.array(xm) % len_th
    xn_mod = -np.array(xn) % len_ph
    if deriv_order == 0:
        coefs_mod = coefs
    elif deriv_order == 1:
        if deriv_dir == "th":
            coefs_mod = 1j * xm * coefs
        elif deriv_dir == "ph":
            coefs_mod = -1j * xn * coefs
    np.add.at(coef_array, np.s_[:, xm_mod, xn_mod], coefs_mod)
    return coef_array


def invert_fourier(
    coefs_nm, xm, xn, len_th, len_ph, deriv_order=0, deriv_dir="", kind=""
):
    coef_array = make_coef_array(
        coefs_nm, xm, xn, len_th, len_ph, deriv_order, deriv_dir
    )
    if kind == "cos":
        return np.fft.ifft2(coef_array, norm="forward").real
    elif kind == "sin":
        return np.fft.ifft2(coef_array, norm="forward").imag
    else:
        return np.fft.ifft2(coef_array, norm="forward")
