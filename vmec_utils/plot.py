import matplotlib.pyplot as plt
import numpy as np
from .fn import normalize, equal_aspect
from . import Booz
from . import Vmec


def make_figax_3d(**kwargs):
    return plt.subplots(1, 1, subplot_kw={"projection": "3d"}, **kwargs)


class Plotter:
    def __init__(self, data: Booz):
        self.data = data

    def plot_slice(self, slice=np.s_[:, :, -1], ax=None, **kwargs):
        # print(self.data.xyzs["xs"].shape)
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        ax.plot_surface(
            self.data.xyzs["xs"][slice],
            self.data.xyzs["ys"][slice],
            self.data.xyzs["zs"][slice],
            rstride=1,
            cstride=1,
            antialiased=True,
            **kwargs,
        )
        equal_aspect(ax)

    def plot_surf(self, ax=None, s_idx=-1, quantity=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        if quantity is None:
            ax.plot_surface(
                self.data.xyzs["xs"][s_idx],
                self.data.xyzs["ys"][s_idx],
                self.data.xyzs["zs"][s_idx],
                rstride=1,
                cstride=1,
                antialiased=True,
                **kwargs,
            )
        else:
            self.data.get_vars(s_idx=s_idx)
            tmp = self.data.vars[quantity][0]
            ax.plot_surface(
                self.data.xyzs["xs"][s_idx],
                self.data.xyzs["ys"][s_idx],
                self.data.xyzs["zs"][s_idx],
                rstride=1,
                cstride=1,
                cmap=plt.cm.jet,
                linewidth=0,
                antialiased=False,
                facecolors=plt.cm.jet(normalize(tmp)),
                **kwargs,
            )
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        equal_aspect(ax)
        return ax

    def plot_cut(self, ax=None, s_idx=None, quantity=None, phi=0):
        phi_idx = np.argmin(np.abs(self.data.ph - phi * np.pi / 180))
        print(phi_idx)
        if s_idx is None:
            s_idx = np.array(
                (np.linspace(0, 1, 7) ** 2) * (len(self.data.s) - 1), dtype=int
            )
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if quantity == None:
            for si in list(s_idx):
                self.data.get_vars(s_idx=si)
                rs = self.data.vars["R"]
                zs = self.data.vars["Z"]
                if si == 0:
                    ax.plot(rs[0, phi_idx], zs[0, phi_idx], "+r")
                ax.plot(rs[:, phi_idx], zs[:, phi_idx], "k")
        #   TODO:
        # else:
        #     for si in list(s_idx):
        #         rs = self.get_var('R', si)[0]
        #         zs = self.get_var('Z', si)[0]
        #         q  = self.get_var(quantity, si)[0]
        #         if si==0:
        #             ax.plot(rs[0,phi_idx], zs[0,phi_idx], '+r')
        #         ax.plot(rs[:,phi_idx], zs[:,phi_idx], 'k')
        ax.set(
            xlabel="R [m]",
            ylabel="Z [m]",
            title=rf"$\varphi$={180*self.data.ph[phi_idx]/np.pi:.2f}º",
        )
        ax.set_aspect("equal")
        return ax

    def plot_profile(self, var, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.data.s, self.data.woutdata[var], **kwargs)

    def plot_axis(self, ax=None):
        xyzs = self.data.get_axis()
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        ax.plot(*xyzs, "r")
        equal_aspect(ax)
        return ax

    def plot_vecbase(self, idth, idph, ax, s_idx=-1, **kwargs):
        xs = self.data.xyzs["xs"][s_idx]
        ys = self.data.xyzs["ys"][s_idx]
        zs = self.data.xyzs["zs"][s_idx]
        ax.quiver(
            xs[idth, idph],
            ys[idth, idph],
            zs[idth, idph],
            *self.data.vecs["e_th"][:, s_idx, idth, idph],
            color="r",
            **kwargs,
        )
        ax.quiver(
            xs[idth, idph],
            ys[idth, idph],
            zs[idth, idph],
            *self.data.vecs["e_ph"][:, s_idx, idth, idph],
            color="g",
            **kwargs,
        )
        ax.quiver(
            xs[idth, idph],
            ys[idth, idph],
            zs[idth, idph],
            *self.data.vecs["e_s"][:, s_idx, idth, idph],
            color="b",
            **kwargs,
        )
        ax.quiver(
            xs[idth, idph],
            ys[idth, idph],
            zs[idth, idph],
            *self.data.vecs["grad_th"][:, s_idx, idth, idph],
            color="r",
            ls="--",
            **kwargs,
        )
        ax.quiver(
            xs[idth, idph],
            ys[idth, idph],
            zs[idth, idph],
            *self.data.vecs["grad_ph"][:, s_idx, idth, idph],
            color="g",
            ls="--",
            **kwargs,
        )
        ax.quiver(
            xs[idth, idph],
            ys[idth, idph],
            zs[idth, idph],
            *self.data.vecs["grad_s"][:, s_idx, idth, idph],
            color="b",
            ls="--",
            **kwargs,
        )
