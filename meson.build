project(
  'vmec_utils',
  'c',
  default_options: ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(
  py,
  [
    '-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())',
  ],
  check: true,
).stdout().strip()

incdir_f2py = run_command(
  py,
  [
    '-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
  ],
  check: true,
).stdout().strip()

# CXX = meson.get_compiler('c')
# libs_you_need_to_link = ['openmp', 'gomp']
# deps = []
# foreach lib_name : libs_you_need_to_link
#     deps += CXX.find_library(lib_name, dirs : [meson.current_source_dir() + '/lib'])
# endforeach

openmp = dependency('openmp')
# lgomp =  meson.get_compiler('c').find_library('gomp')

FORTRAN_MODULE_NAME = 'vmec_helper'
f2py_target = custom_target(
  FORTRAN_MODULE_NAME + '.c',
  input: ['vmec_utils/vmec_helper/vmec_helper.f90'],
  output: [
    FORTRAN_MODULE_NAME + 'module.c',
    #   FORTRAN_MODULE_NAME+'-f2pywrappers.f',
    FORTRAN_MODULE_NAME
    + '-f2pywrappers2.f90',
  ],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', FORTRAN_MODULE_NAME, '--lower'],
)

inc_np = include_directories(incdir_numpy, incdir_f2py)

py.extension_module(
  FORTRAN_MODULE_NAME,
  ['vmec_utils/vmec_helper/vmec_helper.f90', f2py_target],
  incdir_f2py / 'fortranobject.c', # f2py file
  include_directories: inc_np,
  dependencies: [py_dep, openmp],
  install: true,
  subdir: meson.project_name(), # Directory where it will be installed inside site-packages
)

# IMPORTANT: all files must be tracked manually for installation.
# IMPORTANT: directory structure is NOT mantained - It must be tracked also.
py.install_sources(
  [
    'vmec_utils/__init__.py',
    'vmec_utils/class_booz.py',
    'vmec_utils/class_vmec.py',
    'vmec_utils/commands.py',
    'vmec_utils/fn.py',
    'vmec_utils/mercier.py',
    'vmec_utils/plot.py',
    'vmec_utils/utils.py',
  ],
  subdir: meson.project_name(), # Directory where it will be installed inside site-packages
)
