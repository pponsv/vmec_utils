# %%

import matplotlib.pyplot as plt
import numpy as np
import vmec_utils as vl
from time import time


def make_time(fun, *args, **kwargs):
    t0 = time()
    out = fun(*args, **kwargs)
    t1 = time()
    print(fun.__name__, "- Took: ", t1 - t0, "s")
    return out


def make_test(vfile, th, ph):
    tmp = vl.Booz(vfile, theta=th, phi=ph)
    # tmp_sadig = vl.class_vmec_sadig.vmec(vfile, theta=th, phi=ph, s=tmp.s)
    #   Get vars and vecs
    make_time(tmp.get_vars)
    # make_time(tmp.get_vecs)
    # #   Test Rs
    # b = tmp.vars["R"]
    # a = make_time(tmp_sadig.get_r)
    # print("Test R: ", np.all(np.isclose(a, b)))
    # #   Test Zs
    # b = tmp.vars["Z"]
    # a = make_time(tmp_sadig.get_z)
    # print("Test Z: ", np.all(np.isclose(a, b)))
    # #   Test lambda
    # b = tmp.vars["lambda"]
    # a = make_time(tmp_sadig.get_lambda)
    # print("Test lambda: ", np.all(np.isclose(a, b)))
    # #   Test jac
    # b = tmp.vars["sqrt_g"]
    # a = tmp_sadig.get_sqrtg()
    # print("Test jac: ", np.all(np.isclose(a, -b)))
    # #   Test modb
    # b = tmp.vars["mod_b"]
    # a = tmp_sadig.get_B()
    # print("Test modb: ", np.all(np.isclose(a, b)))
    # #   Test b_sub_s
    # b = tmp.vars["b_sub_s"]
    # a = tmp_sadig.get_bsubs()
    # print("Test b_sub_s: ", np.all(np.isclose(a, b)))
    # #   Test b_sub_u
    # b = tmp.vars["b_sub_u"]
    # a = tmp_sadig.get_bsubu()
    # print("Test b_sub_u: ", np.all(np.isclose(a, b)))
    # #   Test b_sub_v
    # b = tmp.vars["b_sub_v"]
    # a = tmp_sadig.get_bsubv()
    # print("Test b_sub_v: ", np.all(np.isclose(a, b)))
    # #   Test b_sup_ubooz


# print("Test b_sup_u: ", np.all(np.isclose(a, b)))
# #   Test b_sup_v
# b = tmp.vars["b_sup_v"]
# a = tmp_sadig.get_bsupv()
# print("Test b_sup_v: ", np.all(np.isclose(a, b)))


# %%
#   Initialize

vfile = "/home/pedro/Documents/tmp/vmecs/tjii_0.10/wout_tj2_eccdw_01.nc"
bfile = "/home/pedro/Documents/tmp/vmecs/tjii_0.10/boozmn_tj2_eccdw_01.nc"

thetas = np.linspace(0, 2 * np.pi, 20)
phis = np.linspace(0, 0.25 * np.pi, 25)

# vmec = vl.Vmec(vfile, theta=thetas, phi=phis)
booz = vl.Booz(bfile, theta=thetas, phi=phis)

# vmec.get_vars()
# booz.get_vars()
quantity = None
# print(vmec.s[-10], booz.s[-1])
# vmec_plotter = vl.Plotter(vmec)
# vmec_plotter.plot_surf(quantity=quantity, s_idx=-11)

s_idx = -1

booz.get_vectors()
print(booz.s_vmec[s_idx])

# %%

booz_plotter = vl.Plotter(booz)
# booz.get_xyzs(s_idx=-1)
booz.get_xyzs()
# ax = booz_plotter.plot_surf(quantity="mod_b", s_idx=s_idx, alpha=0.1)
fig, ax = vl.plot.make_figax_3d()

idph = 0
print(booz.xyzs["xs"].shape)
booz_plotter.plot_slice(np.s_[:, :, idph], ax=ax, color="b", alpha=0.1)
booz_plotter.plot_slice(np.s_[:, 4, :], ax=ax, color="g", alpha=0.1)
booz_plotter.plot_slice(np.s_[-1, :, :], ax=ax, color="gray", alpha=0.1)
for idph in [0, 5]:
    for idth in range(0, len(booz.th)):
        booz_plotter.plot_vecbase(
            idth, idph, ax=ax, length=0.05, s_idx=-1, normalize=True
        )


plt.show()

# %%

jac_sub = np.einsum(
    "ijkl,ijkl->jkl",
    booz.vecs["e_s"],
    np.cross(booz.vecs["e_th"], booz.vecs["e_ph"], axisa=0, axisb=0, axisc=0),
)
jac_sup = 1 / np.einsum(
    "ijkl,ijkl->jkl",
    booz.vecs["grad_s"],
    np.cross(booz.vecs["grad_th"], booz.vecs["grad_ph"], axisa=0, axisb=0, axisc=0),
)
print(np.max(jac_sub), np.min(jac_sub))
print(np.max(jac_sup), np.min(jac_sup))
print(np.all(np.isclose(jac_sub, jac_sup)))
print(np.max(booz.vars["sqrt_g"]), np.min(booz.vars["sqrt_g"]))
#   Test
# make_test(vfile, thetas, phis)

#   Pruebasq

# tmp       = vl.vmec(vfile, theta=thetas, phi=phis)
# tmp_sadig = vl.class_vmec_sadig.vmec(vfile, theta=thetas, phi=phis,s=tmp.s)

# tmp.plot_surf(quantity='sqrt_g')
# plt.show()
# ms = tmp.vmec['xm']
# ns = tmp.vmec['xn']
# rmnc = tmp.vmec['rmnc']
# zmnc = tmp.vmec['zmns']
# th = tmp.th
# ph = tmp.ph
# s  = tmp.s

# e_s = tmp_sadig.get_bsubs()
# tmp.get_vars()
# tmp.get_vecs()

# vecs_s = tmp_sadig.get_vectors() #cambio en el source para que return drds!!
# vecs_p = vl.fn.timer(tmp.get_vecs)
# tmp    = vl.fn.timer(vl.vh.dgen_ds, tmp.vmec['rmnc'].T, tmp.s, tmp.th, tmp.ph, ms, ns, typ='c')


# print(np.all(np.isclose(vecs_s, vecs_p, rtol=1e-5)))
# print(np.max(np.abs(vecs_s-vecs_p)))

# vl.fn.timer(tmp.get_vars)

# b = vl.fn.timer(vl.vh.genvar_modi, rmnc.T, thetas, phis, ms, ns, ms, 'c')

# a = vl.fn.timer(vl.vmec_math.mode_costransform, thetas, -phis, rmnc, ms, ns, ms)

# print(np.all(np.isclose(a, b)))
# print(np.all(np.isclose(b, out)))
# idx = 0
# print(np.all(np.isclose(vecs_s[idx], vecs_p[idx], rtol=1e-7)))
# plt.figure()
# plt.imshow(vecs_s[idx] - vecs_p[idx])
# plt.colorbar()
# plt.figure()
# plt.imshow(vecs_s[idx])
# plt.colorbar()
# plt.figure()
# plt.imshow(vecs_p[idx])
# plt.colorbar()
# # plt.figure()
# # plt.imshow(a[-1])
# plt.show()

# a = np.random.rand(1000)
# b = vh.vmec_helper.prueba(a)
# c = np.sin(a)

# print(np.all(np.isclose(b, c)))

# %%
