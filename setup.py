from distutils.core import setup, Extension
import os

from numpy.distutils.misc_util import *

numpyincl = get_numpy_include_dirs()

cvbldamodule = Extension("cvbLDA",
                    sources = ["cvbLDA.c"],
                    include_dirs = [os.getcwd()] + numpyincl,
                    library_dirs = [],
                    libraries = [],
                    extra_compile_args = ['-O3','-Wall'],
                    extra_link_args = [])

setup(name = 'cvbLDA',
      description = 'Collapsed Variational Bayesian inference for LDA',
      version = '0.1.1',
      author = 'David Andrzejewski',
      author_email = 'andrzeje@cs.wisc.edu',
      license = 'GNU General Public License (Version 3 or later)',
      url = 'http://pages.cs.wisc.edu/~andrzeje/research/cvb_lda.html',
      ext_modules = [cvbldamodule],
      py_modules = [])
