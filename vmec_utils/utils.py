import numpy as np
import matplotlib.pyplot as plt

from .fn import equal_aspect

from .plot import make_figax_3d

from .class_booz import Booz
from .class_vmec import Vmec


def make_coef_array(
    coefs, xm, xn, len_th, len_ph, deriv_order=0, deriv_dir=""
):
    """Make a 3D array of coefficients with the correct shape to be inverted."""
    coef_array = np.zeros(
        (coefs.shape[0], len_th, len_ph), dtype=np.complex128
    )
    if deriv_order == 0:
        coef_array[:, xm, -xn] = coefs
    elif deriv_order == 1:
        if deriv_dir == "th":
            coef_array[:, xm, -xn] = 1j * xm * coefs
        elif deriv_dir == "zt":
            coef_array[:, xm, -xn] = -1j * xn * coefs
        else:
            raise ValueError("Invalid deriv_dir")
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


def get_xyz(rs, phs, zs, slc=np.s_[:]):
    return (
        rs[slc] * np.cos(phs[slc]),
        rs[slc] * np.sin(phs[slc]),
        zs[slc],
    )


def preview_booz(booz_file):
    booz = Booz(booz_file)
    th = np.linspace(0, 2 * np.pi, 100)
    ph = np.linspace(0, 2 * np.pi, 400)
    s = booz.s_vmec

    rs = invert_fourier(
        booz.woutdata["rmnc_b"][-1][np.newaxis, :],
        booz.woutdata["ixm_b"],
        booz.woutdata["ixn_b"],
        len(th),
        len(ph),
        kind="cos",
    )
    zs = invert_fourier(
        booz.woutdata["zmns_b"][-1][np.newaxis, :],
        booz.woutdata["ixm_b"],
        booz.woutdata["ixn_b"],
        len(th),
        len(ph),
        kind="sin",
    )
    ps = invert_fourier(
        booz.woutdata["pmns_b"][-1][np.newaxis, :],
        booz.woutdata["ixm_b"],
        booz.woutdata["ixn_b"],
        len(th),
        len(ph),
        kind="sin",
    )
    phi_cyl = ph + ps

    xyz = get_xyz(rs, phi_cyl, zs)
    x, y, z = xyz

    fig = plt.figure(figsize=[7, 9])

    ax_3d = fig.add_subplot(221, projection="3d")
    ax_3d.plot_surface(
        x[0],
        y[0],
        z[0],
        alpha=0.5,
        color="orange",
        # rstride=1,
        # cstride=1,
    )
    ax_3d.set(title="LCFS")

    ax_polcut = fig.add_subplot(222, projection="3d")
    # ax_polcut.plot(x[0, :, 0], z[0, :, 0])
    booz_plot_cut(booz_file, ax_polcut, phi_deg=-45, num_contours=5)
    ax_polcut.set_aspect("equal")
    ax_polcut.set(title="Poloidal Cut")

    ax_iota = fig.add_subplot(223)
    ax_iota.plot(s[1:], booz.woutdata["iota_b"][1:])
    ax_iota.set(title="Iota")

    ax_pres = fig.add_subplot(224)
    ax_pres.plot(s, booz.woutdata["pres_b"])
    ax_pres.set(title="Pressure")

    equal_aspect(ax_3d)


def preview_vmec(vmec_file):
    vmec = Vmec(vmec_file)
    th = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    s = vmec.s

    xm = vmec.woutdata["xm"].astype(int)
    xn = vmec.woutdata["xn"].astype(int)
    print(xm, xn)

    rs = invert_fourier(
        vmec.woutdata["rmnc"][-1][np.newaxis, :],
        xm,
        xn,
        len(th),
        len(ph),
        kind="cos",
    )
    zs = invert_fourier(
        vmec.woutdata["zmns"][-1][np.newaxis, :],
        xm,
        xn,
        len(th),
        len(ph),
        kind="sin",
    )
    phi_cyl = ph

    xyz = get_xyz(rs, phi_cyl, zs)
    x, y, z = xyz

    fig = plt.figure(figsize=[7, 9])

    ax_3d = fig.add_subplot(321, projection="3d")
    ax_3d.plot_surface(
        x[0],
        y[0],
        z[0],
        alpha=0.5,
        color="orange",
        # rstride=1,
        # cstride=1,
    )
    ax_3d.set(title="LCFS")

    ax_polcut = fig.add_subplot(322)
    vmec_plot_cut(vmec_file, ax=ax_polcut, phi_deg=-45, num_contours=5)
    ax_polcut.set_aspect("equal")
    ax_polcut.set(title="Poloidal Cut")

    ax_iota = fig.add_subplot(323)
    ax_iota.plot(s[1:], vmec.woutdata["iotas"][1:])
    ax_iota.set(title="Iota")

    ax_pres = fig.add_subplot(324)
    ax_pres.plot(s, vmec.woutdata["mass"])
    ax_pres.plot(s, vmec.woutdata["pres"])
    ax_pres.set(title="Pressure - Density")

    ax_beta = fig.add_subplot(325)
    ax_beta.plot(s, vmec.woutdata["beta_vol"])
    ax_beta.set(title="Beta")

    ax_jcurr = fig.add_subplot(326)
    ax_jcurr.plot(s, vmec.woutdata["jdotb"])
    ax_jcurr.set(title="JdotB")

    equal_aspect(ax_3d)
    plt.tight_layout()


def roll_theta(arr):
    """
    adds the first row in the second axis to the end of the array
    """
    return np.append(arr, arr[:, 0, np.newaxis], axis=1)


def vmec_plot_cut(vmec_file, ax=None, phi_deg=0, num_contours=10, **kwargs):
    vmec = Vmec(vmec_file)
    th = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ph_idx = np.argmin(np.abs(ph - np.deg2rad(phi_deg) % (2 * np.pi)))
    print(ph_idx, np.deg2rad(phi_deg))
    s = vmec.s

    xm = vmec.woutdata["xm"].astype(int)
    xn = vmec.woutdata["xn"].astype(int)
    print(xm, xn)

    rs = invert_fourier(
        vmec.woutdata["rmnc"],
        xm,
        xn,
        len(th),
        len(ph),
        kind="cos",
    )
    zs = invert_fourier(
        vmec.woutdata["zmns"],
        xm,
        xn,
        len(th),
        len(ph),
        kind="sin",
    )

    plot_kwargs = dict(c="k", alpha=0.6)
    plot_kwargs.update(kwargs)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    # roll_theta for the plot to be closed, but not before to avoid double plotting
    ax.plot(rs[:, ::10, ph_idx], zs[:, ::10, ph_idx], lw=1, **plot_kwargs)
    rs = roll_theta(rs)
    zs = roll_theta(zs)
    for i in np.arange(rs.shape[0] - 1, 0, -rs.shape[0] // num_contours):
        ax.plot(rs[i, :, ph_idx], zs[i, :, ph_idx], **plot_kwargs)
    ax.set_aspect("equal")


def booz_plot_cut(booz_file, ax=None, phi_deg=0, num_contours=10, **kwargs):
    booz = Booz(booz_file)
    th = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ph_idx = np.argmin(np.abs(ph - np.deg2rad(phi_deg) % (2 * np.pi)))
    print(ph_idx, np.deg2rad(phi_deg))
    s = booz.s_vmec

    rs = invert_fourier(
        booz.woutdata["rmnc_b"],
        booz.woutdata["ixm_b"],
        booz.woutdata["ixn_b"],
        len(th),
        len(ph),
        kind="cos",
    )
    zs = invert_fourier(
        booz.woutdata["zmns_b"],
        booz.woutdata["ixm_b"],
        booz.woutdata["ixn_b"],
        len(th),
        len(ph),
        kind="sin",
    )
    ps = invert_fourier(
        booz.woutdata["pmns_b"],
        booz.woutdata["ixm_b"],
        booz.woutdata["ixn_b"],
        len(th),
        len(ph),
        kind="sin",
    )
    phi_cyl = ph + ps
    xyz = get_xyz(rs, phi_cyl, zs)
    x, y, z = xyz
    print(x.shape)
    plot_kwargs = dict(c="k", alpha=0.6)
    plot_kwargs.update(kwargs)
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
    # roll_theta for the plot to be closed, but not before to avoid double plotting
    for idx in range(x.shape[1] // 10):
        ax.plot(
            x[:, 10 * idx, ph_idx],
            y[:, 10 * idx, ph_idx],
            z[:, 10 * idx, ph_idx],
            lw=1,
            **plot_kwargs
        )
    x = roll_theta(x)
    y = roll_theta(y)
    z = roll_theta(z)
    for i in np.arange(rs.shape[0] - 1, 0, -rs.shape[0] // num_contours):
        ax.plot(
            x[i, :, ph_idx], y[i, :, ph_idx], z[i, :, ph_idx], **plot_kwargs
        )
    ax.set_aspect("equal")
