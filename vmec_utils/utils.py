import numpy as np
import matplotlib.pyplot as plt

from .fn import equal_aspect

from .plot import make_figax_3d

from .class_booz import Booz
from .class_vmec import Vmec


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
        x[0], y[0], z[0], alpha=0.5, color="orange", rstride=1, cstride=1
    )
    ax_3d.set(title="LCFS")

    ax_polcut = fig.add_subplot(222)
    ax_polcut.plot(x[0, :, 0], z[0, :, 0])
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
    th = np.linspace(0, 2 * np.pi, 100)
    ph = np.linspace(0, 2 * np.pi, 400)
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
    # ps = invert_fourier(
    #     vmec.woutdata["pmns"][-1][np.newaxis, :],
    #     xm,
    #     xn,
    #     len(th),
    #     len(ph),
    #     kind="sin",
    # )
    phi_cyl = ph

    xyz = get_xyz(rs, phi_cyl, zs)
    x, y, z = xyz

    fig = plt.figure(figsize=[7, 9])

    ax_3d = fig.add_subplot(321, projection="3d")
    ax_3d.plot_surface(
        x[0], y[0], z[0], alpha=0.5, color="orange", rstride=1, cstride=1
    )
    ax_3d.set(title="LCFS")

    ax_polcut = fig.add_subplot(322)
    ax_polcut.plot(x[0, :, 0], z[0, :, 0])
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
