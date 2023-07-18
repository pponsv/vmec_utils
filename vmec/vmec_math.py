import numpy as np

"""
COMENTARIO:

Sadig ha hecho el cambio de variable:
    n' = -n
De forma que calcula:
    cos(m*th + n'*ph) = cos(m*th)cos(n'*ph) - sin(m*th)*sin(n'*ph)
    sin(m*th + n'*ph) = sin(m*th)cos(n'*ph) + cos(m*th)*sin(n'*ph)

Esto es autoconsistente en su código (hasta donde yo sé)
"""


def costransform(theta, phi, qmnc, xm, xn):
    ns = qmnc.shape[0]
    xmt = xm * theta[:, None]
    xnz = xn * phi[:, None]
    f = np.zeros((ns, theta.size, phi.size))
    for i in range(ns):
        f[i, :, :] = np.matmul(qmnc[i, :] * np.cos(xmt), np.cos(xnz).T) - np.matmul(
            qmnc[i, :] * np.sin(xmt), np.sin(xnz).T
        )
    return f


def sintransform(theta, phi, qmns, xm, xn):
    ns = qmns.shape[0]
    xmt = xm * theta[:, None]
    xnz = xn * phi[:, None]
    f = np.zeros((ns, theta.size, phi.size))
    for i in range(ns):
        f[i, :, :] = np.matmul(qmns[i, :] * np.sin(xmt), np.cos(xnz).T) + np.matmul(
            qmns[i, :] * np.cos(xmt), np.sin(xnz).T
        )
    return f


def mode_costransform(theta, phi, qmnc, xm, xn, mode):
    ns = qmnc.shape[0]
    xmt = xm * theta[:, None]
    xnz = xn * phi[:, None]
    f = np.zeros((ns, theta.size, phi.size))
    mode = mode * np.ones(theta.size)[:, None]
    for i in range(ns):
        f[i, :, :] = np.matmul(
            mode * qmnc[i, :] * np.cos(xmt), np.cos(xnz).T
        ) - np.matmul(mode * qmnc[i, :] * np.sin(xmt), np.sin(xnz).T)
    return f


def mode_sintransform(theta, phi, qmns, xm, xn, mode):
    ns = qmns.shape[0]
    xmt = xm * theta[:, None]
    xnz = xn * phi[:, None]
    f = np.zeros((ns, theta.size, phi.size))
    mode = mode * np.ones(theta.size)[:, None]
    for i in range(ns):
        f[i, :, :] = np.matmul(
            mode * qmns[i, :] * np.sin(xmt), np.cos(xnz).T
        ) + np.matmul(mode * qmns[i, :] * np.cos(xmt), np.sin(xnz).T)
    return f


def radially_interpolated_quantity(quantity, s):

    try:
        s_size = s.size
    except:
        s_size = 1

    if s_size > 1:
        return quantity
    elif s == 1:
        quantity_interp = quantity[-1, :, :]
        return quantity_interp
    elif s == 0.0:
        quantity_interp = quantity[0, 0, :]
        return quantity_interp

    ns = quantity.shape[0]
    Psi_N = np.linspace(0, 1, ns)
    for k in range(ns - 1):
        if s < Psi_N[k + 1] and s >= Psi_N[k]:
            i_s = k

    quantity_interp = quantity[i_s, :, :] + (s - Psi_N[i_s]) / (
        Psi_N[i_s + 1] - Psi_N[i_s]
    ) * (quantity[i_s + 1, :, :] - quantity[i_s, :, :])

    return quantity_interp.reshape((1, quantity.shape[1], quantity.shape[2]))


def toroidally_interpolated_quantity(quantity, phi):

    if phi == 0.0:
        quantity_interp = quantity[:, :, 0]
        return quantity_interp
    elif phi == 2 * np.pi:
        quantity_interp = quantity[:, :, -1]
        return quantity_interp

    print(quantity.shape)
    nphi = quantity.shape[2]
    Phi_N = np.linspace(0, 2 * np.pi, nphi)
    for k in range(nphi - 1):
        if phi < Phi_N[k + 1] and phi >= Phi_N[k]:
            i_phi = k

    quantity_interp = quantity[:, :, i_phi] + (phi - Phi_N[i_phi]) / (
        Phi_N[i_phi + 1] - Phi_N[i_phi]
    ) * (quantity[:, :, i_phi + 1] - quantity[:, :, i_phi])

    return quantity_interp.reshape((quantity.shape[0], quantity.shape[1], 1))


def dotprod_car(x, y):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def dotprod_cyl(x, y, r):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def crossprod(x, y):
    return np.array(
        [
            x[1] * y[2] - x[2] * y[1],
            -x[0] * y[2] + x[2] * y[0],
            x[0] * y[1] - x[1] * y[0],
        ]
    )


def FinDif2nd(ypoints, xpoints, points):

    interp = np.zeros(ypoints.shape)
    derinterp = np.zeros(ypoints.shape)
    dx = xpoints[1] - xpoints[0]
    for k in range(1, ypoints.shape[0] - 1):
        interp[k] = (
            ypoints[k - 1]
            * (points[k] - xpoints[k])
            * (points[k] - xpoints[k + 1])
            / (2 * dx**2)
            - ypoints[k]
            * (points[k] - xpoints[k - 1])
            * (points[k] - xpoints[k + 1])
            / (dx**2)
            + ypoints[k + 1]
            * (points[k] - xpoints[k - 1])
            * (points[k] - xpoints[k])
            / (2 * dx**2)
        )
        derinterp[k] = (
            ypoints[k - 1]
            * ((points[k] - xpoints[k]) + (points[k] - xpoints[k + 1]))
            / (2 * dx**2)
            - ypoints[k]
            * ((points[k] - xpoints[k - 1]) + (points[k] - xpoints[k + 1]))
            / (dx**2)
            + ypoints[k + 1]
            * ((points[k] - xpoints[k - 1]) + (points[k] - xpoints[k]))
            / (2 * dx**2)
        )

    interp[0] = (
        ypoints[0] * (points[0] - xpoints[1]) * (points[0] - xpoints[2]) / (2 * dx**2)
        - ypoints[1] * (points[0] - xpoints[0]) * (points[0] - xpoints[2]) / (dx**2)
        + ypoints[2]
        * (points[0] - xpoints[0])
        * (points[0] - xpoints[1])
        / (2 * dx**2)
    )
    derinterp[0] = (
        ypoints[0]
        * ((points[0] - xpoints[1]) + (points[0] - xpoints[2]))
        / (2 * dx**2)
        - ypoints[1] * ((points[0] - xpoints[0]) + (points[0] - xpoints[2])) / (dx**2)
        + ypoints[2]
        * ((points[0] - xpoints[0]) + (points[0] - xpoints[1]))
        / (2 * dx**2)
    )
    interp[-1] = (
        ypoints[-3]
        * (points[-1] - xpoints[-2])
        * (points[-1] - xpoints[-1])
        / (2 * dx**2)
        - ypoints[-2]
        * (points[-1] - xpoints[-3])
        * (points[-1] - xpoints[-1])
        / (dx**2)
        + ypoints[-1]
        * (points[-1] - xpoints[-3])
        * (points[-1] - xpoints[-2])
        / (2 * dx**2)
    )
    derinterp[-1] = (
        ypoints[-3]
        * ((points[-1] - xpoints[-2]) + (points[-1] - xpoints[-1]))
        / (2 * dx**2)
        - ypoints[-2]
        * ((points[-1] - xpoints[-3]) + (points[-1] - xpoints[-1]))
        / (dx**2)
        + ypoints[-1]
        * ((points[-1] - xpoints[-3]) + (points[-1] - xpoints[-2]))
        / (2 * dx**2)
    )

    return interp, derinterp
