import numpy as np
from scipy.io import netcdf
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants as physcnt
import matplotlib.pyplot as plt
from matplotlib import cm
from . import vmec_math as vmath

# from .vmec_math import *


def plot_surf_3D(
    woutnc="NOVMECFILE.nc",
    thetamin=0,
    thetamax=2 * np.pi,
    phimin=0,
    phimax=2 * np.pi,
    s=1.0,
):

    woutclass = vmec(
        woutnc=woutnc,
        theta=np.linspace(thetamin, thetamax, 100),
        phi=np.linspace(phimin, phimax, 600),
        s=s,
    )
    r = woutclass.get_r()
    z = woutclass.get_z()

    x = r * np.cos(woutclass.phi)
    y = r * np.sin(woutclass.phi)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.grid(False)
    if x.shape[0] == 1:
        x = x[0]
        y = y[0]
        z = z[0]
    ax.plot_surface(x, y, z, color="magenta", alpha=0.3)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    return fig, ax


def plot_vmec_3D(woutnc="NOVMECFILE.nc", quantity="sqrtg", s=1.0):

    woutclass = vmec(
        woutnc=woutnc,
        theta=np.linspace(0, 2 * np.pi, 100),
        phi=np.linspace(0, 2 * np.pi, 600),
        s=s,
    )
    r = woutclass.get_r()
    z = woutclass.get_z()

    if quantity == "sqrtg":
        q = woutclass.get_sqrtg()
    elif quantity == "B":
        q = woutclass.get_B()
    elif quantity == "grads_sqrtg":
        q = woutclass.get_grads_sqrtg()
    else:
        print("quantity " + quantity + " not implemented.\n")
        return 0

    if s != 1.0:
        q = q[0]
        r = r[0]
        z = z[0]

    X = r * np.cos(woutclass.phi)
    Y = r * np.sin(woutclass.phi)
    Z = z

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    V_normalized = q - q.min().min()
    V_normalized = V_normalized / V_normalized.max().max()

    cont = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        cmap=plt.cm.jet,
        linewidth=0,
        antialiased=False,
        facecolors=plt.cm.jet(V_normalized),
    )

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(q)
    fig.colorbar(m, label=quantity)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    return fig, ax


def plot_vmec_2D(woutnc="NOVMECFILE.nc", quantity="sqrtg", s=1.0):

    woutclass = vmec(
        woutnc=woutnc,
        theta=np.linspace(0, 2 * np.pi, 100),
        phi=np.linspace(0, 2 * np.pi, 150),
        s=s,
    )
    if quantity == "sqrtg":
        q = woutclass.get_sqrtg()
    elif quantity == "B":
        q = woutclass.get_B()
    elif quantity == "grads_sqrtg":
        q = woutclass.get_grads_sqrtg()
    else:
        print("quantity " + quantity + " not implemented.\n")
        return 0
    if s != 1.0:
        q = q[0]
    fig, ax = plt.subplots()
    contf = ax.contourf(woutclass.phi, woutclass.theta, q, 50, cmap="jet")
    cbar = fig.colorbar(contf, label=quantity)
    ax.set_xlabel(r"$\phi_v$")
    ax.set_ylabel(r"$\theta_v$")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    return fig, ax


class vmec:
    def __init__(
        self,
        woutnc="NOVMECFILE.nc",
        theta=np.linspace(0, 2 * np.pi, 100),
        phi=np.linspace(0, 2 * np.pi, 150),
        s=1.0,
    ):

        self.file = woutnc
        self.theta = theta
        self.phi = phi
        self.nphi = phi.size
        self.ntheta = theta.size
        self.variables = {}

        f = netcdf.NetCDFFile(self.file, "r")
        for i in f.variables:
            self.variables[i] = f.variables[i].data.copy()
        self.variables["xn"] = -self.variables["xn"]
        self.variables["xn_nyq"] = -self.variables["xn_nyq"]
        f.close()

        if s == "all":
            self.s = np.linspace(0, 1, self.variables["ns"])
            self.ns = self.variables["ns"]
        else:
            self.s = s
            self.ns = 1

    def get_B(self):
        q = vmath.costransform(
            self.theta,
            self.phi,
            self.variables["bmnc"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_bsupu(self):
        q = vmath.costransform(
            self.theta,
            self.phi,
            self.variables["bsupumnc"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_bsupv(self):
        q = vmath.costransform(
            self.theta,
            self.phi,
            self.variables["bsupvmnc"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_bsubs(self):
        q = vmath.sintransform(
            self.theta,
            self.phi,
            self.variables["bsubsmns"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_bsubv(self):
        q = vmath.costransform(
            self.theta,
            self.phi,
            self.variables["bsubvmnc"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_bsubu(self):
        q = vmath.costransform(
            self.theta,
            self.phi,
            self.variables["bsubumnc"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_sqrtg(self):
        q = -vmath.costransform(
            self.theta,
            self.phi,
            self.variables["gmnc"],
            self.variables["xm_nyq"],
            self.variables["xn_nyq"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_grads_sqrtg(self):
        grads = self.get_vectors()["grads"]
        q = self.get_sqrtg() * np.sqrt(grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2)
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_gradv_sqrtg(self):
        gradv = self.get_vectors()["gradv"]
        q = self.get_sqrtg() * np.sqrt(gradv[0] ** 2 + gradv[1] ** 2 + gradv[2] ** 2)
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_r(self):
        q = vmath.costransform(
            self.theta,
            self.phi,
            self.variables["rmnc"],
            self.variables["xm"],
            self.variables["xn"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_z(self):
        q = vmath.sintransform(
            self.theta,
            self.phi,
            self.variables["zmns"],
            self.variables["xm"],
            self.variables["xn"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_lambda(self):
        q = vmath.sintransform(
            self.theta,
            self.phi,
            self.variables["lmns"],
            self.variables["xm"],
            self.variables["xn"],
        )
        return vmath.radially_interpolated_quantity(q, self.s)

    def get_Itor(self):
        bsubu = self.get_bsubu()
        return (
            1
            / (2 * np.pi * physcnt["mag. constant"][0])
            * np.sum(
                bsubu[:, :-1, :-1]
                + bsubu[:, 1:, 1:]
                + bsubu[:, 1:, :-1]
                + bsubu[:, :-1, :-1],
                (1, 2),
            )
            / 4
            * np.diff(self.phi)[0]
            * np.diff(self.theta)[0]
        )

    def get_Ipol_disk(self):
        bsubv = self.get_bsubv()
        return (
            1
            / (2 * np.pi * physcnt["mag. constant"][0])
            * np.sum(
                bsubv[:, :-1, :-1]
                + bsubv[:, 1:, 1:]
                + bsubv[:, 1:, :-1]
                + bsubv[:, :-1, :-1],
                (1, 2),
            )
            / 4
            * np.diff(self.phi)[0]
            * np.diff(self.theta)[0]
        )

    def get_axis(self, phi=None):
        if phi != None:
            s_old = self.s
            phi_old = self.phi
            self.s = 0.0
            self.phi = np.linspace(phi, phi, 1)
            r = self.get_r()
            z = self.get_z()
            self.s = s_old
            self.phi = phi_old
            return r[0], z[0]
        else:
            s_old = self.s
            self.s = 0.0
            r = self.get_r()
            z = self.get_z()
            self.s = s_old
            return r, z

    def get_vectors(self):
        q = {}
        s0 = self.s
        ns0 = self.ns
        self.s = np.linspace(0, 1, self.variables["ns"])
        self.ns = self.variables["ns"]
        r = self.get_r()
        z = self.get_z()
        self.s = s0
        self.ns = ns0
        cosv = np.cos(np.tile(self.phi, (r.shape[0], self.theta.size, 1)))
        sinv = np.sin(np.tile(self.phi, (r.shape[0], self.theta.size, 1)))
        rr, drds = vmath.FinDif2nd(
            r, np.linspace(0, 1, r.shape[0]), np.linspace(0, 1, r.shape[0])
        )
        zz, dzds = vmath.FinDif2nd(
            z, np.linspace(0, 1, z.shape[0]), np.linspace(0, 1, z.shape[0])
        )
        # return drds
        q["esubs"] = np.array([drds * cosv, drds * sinv, dzds])

        q["esubu"] = np.array(
            [
                -vmath.mode_sintransform(
                    self.theta,
                    self.phi,
                    self.variables["rmnc"],
                    self.variables["xm"],
                    self.variables["xn"],
                    self.variables["xm"],
                )
                * cosv,
                -vmath.mode_sintransform(
                    self.theta,
                    self.phi,
                    self.variables["rmnc"],
                    self.variables["xm"],
                    self.variables["xn"],
                    self.variables["xm"],
                )
                * sinv,
                vmath.mode_costransform(
                    self.theta,
                    self.phi,
                    self.variables["zmns"],
                    self.variables["xm"],
                    self.variables["xn"],
                    self.variables["xm"],
                ),
            ]
        )

        q["esubv"] = np.array(
            [
                -vmath.mode_sintransform(
                    self.theta,
                    self.phi,
                    self.variables["rmnc"],
                    self.variables["xm"],
                    self.variables["xn"],
                    self.variables["xn"],
                )
                * cosv
                - r * sinv,
                -vmath.mode_sintransform(
                    self.theta,
                    self.phi,
                    self.variables["rmnc"],
                    self.variables["xm"],
                    self.variables["xn"],
                    self.variables["xn"],
                )
                * sinv
                + r * cosv,
                +vmath.mode_costransform(
                    self.theta,
                    self.phi,
                    self.variables["zmns"],
                    self.variables["xm"],
                    self.variables["xn"],
                    self.variables["xn"],
                ),
            ]
        )

        jac = vmath.dotprod_car(q["esubs"], vmath.crossprod(q["esubu"], q["esubv"]))
        q["grads"] = vmath.crossprod(q["esubu"], q["esubv"]) / jac
        q["gradu"] = vmath.crossprod(q["esubv"], q["esubs"]) / jac
        q["gradv"] = vmath.crossprod(q["esubs"], q["esubu"]) / jac
        for i in q.keys():
            for j in range(0, 3):
                q[i][j] = vmath.radially_interpolated_quantity(q[i][j], self.s)
        self.s = s0
        return q

    def get_metric(self):
        q = {}
        basis = self.get_vectors()
        r = self.get_r()
        q["gsubuu"] = vmath.dotprod_cyl(basis["esubu"], basis["esubu"], 1)
        q["gsubvv"] = vmath.dotprod_cyl(basis["esubv"], basis["esubv"], 1)
        q["gsubss"] = vmath.dotprod_cyl(basis["esubs"], basis["esubs"], 1)
        q["gsubuv"] = vmath.dotprod_cyl(basis["esubu"], basis["esubv"], 1)
        q["gsubus"] = vmath.dotprod_cyl(basis["esubu"], basis["esubs"], 1)
        q["gsubvs"] = vmath.dotprod_cyl(basis["esubv"], basis["esubs"], 1)
        for i in q.keys():
            q[i] = vmath.radially_interpolated_quantity(q[i], self.s)
        return q

    def interp_vmec(self, R, phi, z, q):
        ns = self.ns
        ntheta = self.ntheta
        nphi = self.nphi
        r_vmec = self.get_r()
        z_vmec = self.get_z()
        sqrtg_vmec = self.get_sqrtg()
        phi_vmec = np.tile(self.phi, (ns, ntheta, 1))
        q_vmec = np.zeros(r_vmec.shape)
        interp_q = RegularGridInterpolator([R, phi, z], q)
        q_vmec = interp_q(
            list(zip(r_vmec.flatten(), phi_vmec.flatten(), z_vmec.flatten()))
        ).reshape(ns, ntheta, nphi)
        return r_vmec, z_vmec, sqrtg_vmec, q_vmec
