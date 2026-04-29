project(
  'vmec_utils',
  'c',
  default_options: ['warning_level=2'],
)

add_languages('fortran')

py = import('python').find_installation(pure: false)

numpy_inc = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

f2py_inc = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

inc_np = include_directories(numpy_inc)
inc_f2py = include_directories(f2py_inc)

openmp = dependency('openmp')

FORTRAN_MODULE_NAME = 'vmec_helper'
f2py_target = custom_target(
  FORTRAN_MODULE_NAME + '.c',
  input: ['vmec_utils/vmec_helper/vmec_helper.f90'],
  output: [
    FORTRAN_MODULE_NAME + 'module.c',
    FORTRAN_MODULE_NAME + '-f2pywrappers2.f90',
  ],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', FORTRAN_MODULE_NAME, '--lower'],
)

py.extension_module(
  FORTRAN_MODULE_NAME,
  ['vmec_utils/vmec_helper/vmec_helper.f90', f2py_target],
  f2py_inc / 'fortranobject.c', # f2py file
  include_directories: [inc_np, inc_f2py],
  dependencies: [py.dependency(), openmp],
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
