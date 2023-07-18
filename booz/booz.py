import numpy as np
from ..commands import make_woutname_vmec
from scipy.optimize import basinhopping

# from scipy.optimize import basinhopping


def get_quantity(file, quant, theta, phi, surf):
    isurf = list(file.variables["jlist"].data).index(surf)
    dict_quants = {
        "r": ["rmnc_b", np.cos],
        "z": ["zmns_b", np.sin],
        "b": ["bmnc_b", np.cos],
        "p": ["pmns_b", np.sin],
    }
    key, func = dict_quants[quant]
    kmn = np.array(file.variables[key].data[isurf])
    ns = np.array(file.variables["ixn_b"].data)
    ms = np.array(file.variables["ixm_b"].data)
    return np.sum(kmn * func(ms * theta - ns * phi))


def get_phitor(file, theta, phi, surf):
    """
    In Hirshman’s note “Transformation from VMEC to Boozer coordinates”,
    the angle difference is defined as
        p = phi_booz - phi_tor
    However in the fortran booz_xform code, a minus sign appears on line
    83 of boozer.f that reverses the sign, so the p quantity saved in
    boozmn_*.nc files is in fact
        p = phi_tor - phi_booz
    So that
        phi_tor = phi_booz + p

    source: https://hiddensymmetries.github.io/booz_xform/theory.html#theory
    """
    p = get_quantity(file, "p", theta, phi, surf)
    return phi + p


def get_xyz(file, theta, phi, surf):
    R = get_quantity(file, "r", theta, phi, surf)
    Z = get_quantity(file, "z", theta, phi, surf)
    phitor = get_phitor(file, theta, phi, surf)
    return np.array([R * np.cos(phitor), R * np.sin(phitor), Z])


def posdifsq(file, theta, phi, surf, xyz_0):
    xyz_b = get_xyz(file, theta, phi, surf)
    return np.sum((xyz_b - xyz_0) ** 2)


def getpoints(file, surf, thetas, phis):
    xyzs = np.zeros((len(thetas), len(phis), 3))
    for itheta, theta in enumerate(thetas):
        for iphi, phi in enumerate(phis):
            xyzs[itheta, iphi] = get_xyz(file, theta, phi, surf)
    return xyzs


def findclosest(xyzs, cartpos):
    delta = np.sum((xyzs - cartpos) ** 2, axis=2)
    itheta, iphi = np.unravel_index(np.argmin(delta), delta.shape)
    return itheta, iphi


class MyBounds:
    def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def get_booz_ang_pos(file, surf, xyz_0, init=[0, 0]):
    """
    Finds closest point to coil on the surface _surf_ and gets its magnetic coordinates.
    """
    tfun = lambda thphi: posdifsq(file, *thphi, surf, xyz_0)
    bounds = MyBounds(xmin=[-np.pi, 0], xmax=[np.pi, np.pi * 2])
    res = basinhopping(tfun, init, 100, T=2.0, stepsize=3.0, accept_test=bounds)
    if not res.success:
        print("unsuccessful")
    theta, phi = res.x[0], res.x[1]
    return theta, phi
