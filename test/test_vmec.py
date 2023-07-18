import matplotlib.pyplot as plt
import numpy as np
from plotting_styles import pub_style_one, pub_style_two, rc_context

# import vmec_helper as vh
import vmec_library as vl
from time import time


def make_time(fun, *args, **kwargs):
    t0 = time()
    out = fun(*args, **kwargs)
    t1 = time()
    print(fun.__name__, "- Took: ", t1 - t0, "s")
    return out


def make_test(vfile, th, ph):
    tmp = vl.Vmec(vfile, theta=th, phi=ph)
    tmp_sadig = vl.class_vmec_sadig.vmec(vfile, theta=th, phi=ph, s=tmp.s)
    #   Get vars and vecs
    make_time(tmp.get_vars_old)
    make_time(tmp.get_vecs_old)
    #   Test Rs
    b = tmp.vars["R"]
    a = make_time(tmp_sadig.get_r)
    print("Test R: ", np.all(np.isclose(a, b)))
    #   Test Zs
    b = tmp.vars["Z"]
    a = make_time(tmp_sadig.get_z)
    print("Test Z: ", np.all(np.isclose(a, b)))
    #   Test lambda
    b = tmp.vars["lambda"]
    a = make_time(tmp_sadig.get_lambda)
    print("Test lambda: ", np.all(np.isclose(a, b)))
    #   Test jac
    b = tmp.vars["sqrt_g"]
    a = tmp_sadig.get_sqrtg()
    print("Test jac: ", np.all(np.isclose(a, -b)))
    #   Test modb
    b = tmp.vars["mod_b"]
    a = tmp_sadig.get_B()
    print("Test modb: ", np.all(np.isclose(a, b)))
    #   Test b_sub_s
    b = tmp.vars["b_sub_s"]
    a = tmp_sadig.get_bsubs()
    print("Test b_sub_s: ", np.all(np.isclose(a, b)))
    #   Test b_sub_u
    b = tmp.vars["b_sub_u"]
    a = tmp_sadig.get_bsubu()
    print("Test b_sub_u: ", np.all(np.isclose(a, b)))
    #   Test b_sub_v
    b = tmp.vars["b_sub_v"]
    a = tmp_sadig.get_bsubv()
    print("Test b_sub_v: ", np.all(np.isclose(a, b)))
    #   Test b_sup_u
    b = tmp.vars["b_sup_u"]
    a = tmp_sadig.get_bsupu()
    print("Test b_sup_u: ", np.all(np.isclose(a, b)))
    #   Test b_sup_v
    b = tmp.vars["b_sup_v"]
    a = tmp_sadig.get_bsupv()
    print("Test b_sup_v: ", np.all(np.isclose(a, b)))


if __name__ == "__main__":
    #   Initialize

    vfile = "/home/pedro/MEGA/00_doctorado/research/VMEC/TJ-II/100_44_64.0.0/wout_100_44_64_0.0.nc"

    thetas = np.linspace(0, 2 * np.pi, 100)
    phis = np.linspace(0, 2 * np.pi, 400)

    tmp = vl.Vmec(vfile, theta=thetas, phi=phis)
    make_time(tmp.get_var_old, "R")

    ms = tmp.woutdata["xm"]
    ns = tmp.woutdata["xn"]
    rmn = tmp.woutdata["rmnc"]
    print(rmn.shape, ms.shape, ns.shape)

    #   Test
    # make_test(vfile, thetas, phis)

#   Pruebas

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
