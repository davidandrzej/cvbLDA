/**
   cvbLDA - Implementation of Collapsed Variational Bayesian inference (CVB)
   for the Latent Dirichlet Allocation model (LDA)

   Copyright (C) 2009 David Andrzejewski (andrzeje@cs.wisc.edu)
 
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <Python.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <numpy/arrayobject.h>

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_intLDA

// Uniform rand between [0,1] (inclusive)
#define unif() ((double) rand()) / ((double) RAND_MAX)

#define ARGS_OK 0
#define ARGS_BAD 1

// Means and variances for our Gaussian approximations
// of the 'fields' (counts)
typedef struct {
  PyArrayObject* m_njk; // D x T
  PyArrayObject* v_njk; 

  PyArrayObject* m_nkw; // T x W
  PyArrayObject* v_nkw;
  
  PyArrayObject* m_nk; // T
  PyArrayObject* v_nk;
} counts;

typedef struct {
  PyArrayObject* alpha; // 1 x T
  PyArrayObject* alphasum;
  PyArrayObject* beta; // T x W
  PyArrayObject* betasum;
  int T;
} model_params;

typedef struct {
  int D;
  int W;
  int* Md; // Number of unique words in each document
  int** docs_w; // Each array contains word indices for corresponding doc
  int** docs_c; // Each array contains word counts for corresponding doc
  // Variational parameters, each doc associated with (T x Md) matrix
  PyArrayObject** gamma; 
} dataset;

// Structs used by the 'online-Gibbs' gamma init
typedef struct {
  PyArrayObject* nw;
  PyArrayObject* nd;
  PyArrayObject* nw_colsum;
} g_counts;
typedef struct {
  int D;
  int W;  
  int* doclens;
  int** docs;
} g_dataset;

static PyObject* cvbLDA(PyObject *self, PyObject *args, PyObject* keywds);

static int convert_args(PyObject* docs_w_arg, PyObject* docs_c_arg,
                        PyArrayObject* alpha, PyArrayObject* beta,
                        model_params** p_mp, dataset** p_ds);

static counts* given_init(PyObject* gamma_init,
                          model_params* mp, dataset* ds);

static double cvb_infer(model_params* mp, dataset* ds, counts* c);

static PyArrayObject* est_phi(model_params* mp, dataset* ds, counts* c);
static PyArrayObject* est_theta(model_params* mp, dataset* ds, counts* c);

static void count_update(counts* c, dataset* ds,
                         int i, int j, int w,
                         double scale);

// Functions used by the 'online-Gibbs' gamma init
static counts* online_gamma_init(model_params* mp, dataset* ds,int randseed);

static g_counts* gibbs_online_init(model_params* mp, g_dataset* ds);

static g_dataset* convert_docs(dataset* ds);

static PyArrayObject* g_est_phi(model_params* mp, g_dataset* ds, g_counts* c);
static PyArrayObject* g_est_theta(model_params* mp, g_dataset* ds, g_counts* c);

static int mult_sample(double* vals, double sum);

