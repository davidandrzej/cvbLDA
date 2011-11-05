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

#include "cvbLDA.h"

/**
 * This is the exposed method which is called from Python
 */
static PyObject * cvbLDA(PyObject *self, PyObject *args, PyObject* keywds)
{
  // Null-terminated list of arg keywords
  //
  static char *kwlist[] = {"docs_w","docs_c","alpha","beta","gamma_init",
                           "maxiter","convtol",
                           "verbose","randseed",NULL};

  // Required args
  //
  PyObject* docs_w_arg; // List of Lists
  PyObject* docs_c_arg; // List of Lists
  PyArrayObject* alpha; // NumPy Array
  PyArrayObject* beta; // NumPy Array
  // Optional args
  // (and their default values)
  // 
  PyObject* gamma_init = NULL; // List of NumPy Arrays
  int maxiter = 100; // Max number of iterations to do 
  double convtol = -1; // Gamma convergence tolerance
  int verbose = 0; // 1 = verbose output
  int randseed = 194582; // randseed for gamma init (if applicable)

  // Parse function args
  //
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"O!O!O!O!|O!idii",kwlist,
                                  &PyList_Type,&docs_w_arg,
                                  &PyList_Type,&docs_c_arg,
                                  &PyArray_Type,&alpha,
                                  &PyArray_Type,&beta,
                                  &PyList_Type,&gamma_init,
                                  &maxiter,&convtol,
                                  &verbose,&randseed))
    // ERROR - bad args
    return NULL;
 
  // Use args to populate structs
  // (also check for *validity*)
  //
  model_params* mp;
  dataset* ds;
  if(ARGS_BAD == convert_args(docs_w_arg,docs_c_arg,alpha,beta,&mp,&ds))
    {
      // Args bad! Return to Python...error condition should be set
      return NULL;
    }
  
  // Init counts
  int d;
  counts* c = NULL;  
  if(gamma_init == NULL)
    {
      // Do our own online init of gamma
      c = online_gamma_init(mp,ds,randseed);
    }
  else
    {
      // Use the user-supplied initial gamma
      c = given_init(gamma_init,mp,ds);
    }

  // Check results
  if(c == NULL)
    {
      // ERROR - something went wrong with user-supplied init
      Py_DECREF(mp->alphasum);
      Py_DECREF(mp->betasum);
      free(mp);
      for(d = 0; d < ds->D; d++) 
        {
          free(ds->docs_w[d]);
          free(ds->docs_c[d]);
        }
      free(ds->docs_w);
      free(ds->docs_c);
      free(ds->Md);
      free(ds);
      return NULL;
    }
    
  // Iterate until
  // -we've done maxiter iterations
  // OR
  // -max delta change falls below convtol
  //
  int si;
  double delta; // max L1 change in any single \gamma_ij vector
  for(si=0; si < maxiter; si++)
    {
      delta = cvb_infer(mp,ds,c);
      if(verbose == 1)
        {
          printf("Iteration %d of %d, max L1 change in gamma = %f\n",
                 si,maxiter,delta);
        }
      if(delta < convtol)
        break;
    }
  
  // Estimate phi and theta
  PyArrayObject* phi = est_phi(mp, ds, c);
  PyArrayObject* theta = est_theta(mp, ds, c);

  // Put each document's gamma NumPy array into a List
  PyObject* gammalist = PyList_New(ds->D);
  for(d = 0; d < ds->D; d++)
    PyList_SetItem(gammalist,d,(PyObject*) ds->gamma[d]);
  
  // Package phi, theta, and final sample in tuple for return
  PyObject* retval = PyTuple_New(3);

  PyTuple_SetItem(retval,0,(PyObject*) phi);
  PyTuple_SetItem(retval,1,(PyObject*) theta);
  PyTuple_SetItem(retval,2,(PyObject*) gammalist);

  // Do memory cleanup...
  Py_DECREF(c->m_njk);
  Py_DECREF(c->v_njk);
  Py_DECREF(c->m_nkw);
  Py_DECREF(c->v_nkw);
  Py_DECREF(c->m_nk);
  Py_DECREF(c->v_nk);
  free(c);

  Py_DECREF(mp->alphasum);
  Py_DECREF(mp->betasum);
  free(mp);

  for(d = 0; d < ds->D; d++)
    {
      free(ds->docs_w[d]);
      free(ds->docs_c[d]);
    }
  free(ds->docs_w);
  free(ds->docs_c);
  free(ds->Md);
  free(ds->gamma);
  free(ds);
  
  return (PyObject*) retval;
}

/*
 * Update the count matrices from the gamma values
 * 
 * Scale should be proportional to count of w in doc j
 * Can then multiply by -1 to subtract gamma out 
 */
static void count_update(counts* c, dataset* ds, 
                         int i, int j, int w,
                         double scale) 
{
  int k;
  int T = PyArray_DIM(ds->gamma[0],0);
  for(k = 0; k < T; k++)
    {
      // mean contribution, var contribution
      double mc = scale * (*((double*)PyArray_GETPTR2(ds->gamma[j],k,i)));
      double vc = scale * ((*((double*)PyArray_GETPTR2(ds->gamma[j],k,i))) *
                           (1 - (*((double*)PyArray_GETPTR2(ds->gamma[j],k,i)))));
      *((double*)PyArray_GETPTR2(c->m_njk,j,k)) += mc;
      *((double*)PyArray_GETPTR2(c->v_njk,j,k)) += vc;

      *((double*)PyArray_GETPTR2(c->m_nkw,k,w)) += mc;
      *((double*)PyArray_GETPTR2(c->v_nkw,k,w)) += vc;              

      *((double*)PyArray_GETPTR1(c->m_nk,k)) += mc;
      *((double*)PyArray_GETPTR1(c->v_nk,k)) += vc;              
    }
}


/**
 * Init counts and sampler from a user-supplied initial state
 */
static counts* given_init(PyObject* gamma_init,
                          model_params* mp, dataset* ds)
{  
  // Do some init
  //
  int W = ds->W;
  int D = ds->D;
  int T = mp->T;
 
  // Make sure initial sample has correct number of docs
  if(D != PyList_Size(gamma_init))
    {
      // ERROR
      PyErr_SetString(PyExc_RuntimeError,
                      "Number of docs / init gamma mismatch");
      return NULL;
    }

  // Alloc and init count matrices and init sample
  //
  counts* c = (counts*) malloc(sizeof(counts));  

  npy_intp* njkdims = malloc(sizeof(npy_intp)*2);
  njkdims[0] = D;
  njkdims[1] = T;
  c->m_njk = (PyArrayObject*) PyArray_ZEROS(2,njkdims,PyArray_DOUBLE,0);
  c->v_njk = (PyArrayObject*) PyArray_ZEROS(2,njkdims,PyArray_DOUBLE,0);
  free(njkdims);

  npy_intp* nkwdims = malloc(sizeof(npy_intp)*2);
  nkwdims[0] = T;
  nkwdims[1] = W;
  c->m_nkw = (PyArrayObject*) PyArray_ZEROS(2,nkwdims,PyArray_DOUBLE,0);
  c->v_nkw = (PyArrayObject*) PyArray_ZEROS(2,nkwdims,PyArray_DOUBLE,0);
  free(nkwdims);

  npy_intp* nkdims = malloc(sizeof(npy_intp));
  nkdims[0] = T;
  c->m_nk = (PyArrayObject*) PyArray_ZEROS(1,nkdims,PyArray_DOUBLE,0);
  c->v_nk = (PyArrayObject*) PyArray_ZEROS(1,nkdims,PyArray_DOUBLE,0);
  free(nkdims);

  ds->gamma = malloc(sizeof(PyArrayObject*) * D);

  // Now populate mean/var matrices from given init gamma
  //
  int w,ct; // unique word and its count
  int i,j; // [word doc topic] indices
  // For each doc
  for(j = 0; j < D; j++)
    {
      // Copy over gamma for this document
      ds->gamma[j] = (PyArrayObject*) PyList_GetItem(gamma_init,j);

      // For each unique word in the document
      for(i = 0; i < ds->Md[j]; i++) 
        {
          // Word and its count in this document
          w = ds->docs_w[j][i];
          ct = ds->docs_c[j][i];

          // Update counts for each topic
          count_update(c,ds,i,j,w,ct);
        }
    }  
  return c;
}

/**
 * Do a Collapsed Variational Bayesian inference iteration 
 * Return the current value of the variational free energy
 */
static double cvb_infer(model_params* mp, dataset* ds, counts* c)
{
  // Do some init 
  //
  int D = ds->D;
  int T = mp->T;

  // Update gamma parameters
  //
  double max_delta = 0; // max absolute change in gamma (convergence)
  double cur_delta;
  double* old_gamma = malloc(sizeof(double)*T);
  double normsum; // used to normalize gammas

  // Foreach doc in corpus
  //
  int j,i,k;
  for(j = 0; j < D; j++) 
    {

      int doclen = ds->Md[j];
      int* doc_w = ds->docs_w[j];
      int* doc_c = ds->docs_c[j];
      int f = 0;

      // Get this gamma
      PyArrayObject* gamma = ds->gamma[j];

      // For each word in doc
      for(i = 0; i < doclen; i++)
        {      
          // Word index and count
          int wi = doc_w[i];
          int ci = doc_c[i];

          // Subtract these gamma from count mean/var
          count_update(c,ds,i,j,wi,-1*ci);
              	
          // For each topic, update gamma
          normsum = 0;
          for(k = 0; k < T; k++) 
            { 
              // Save old gamma              
              old_gamma[k] = *((double*)PyArray_GETPTR2(gamma,k,i));

              // Huge messy gamma update eqn (is propto, so re-norm at end)
              double alphaval = *((double*)PyArray_GETPTR2(mp->alpha,f,k));
              double betaval = *((double*)PyArray_GETPTR2(mp->beta,k,wi));
              double betasum = *((double*)PyArray_GETPTR1(mp->betasum,k));
               
              double mnjk = *((double*)PyArray_GETPTR2(c->m_njk,j,k));
              double vnjk = *((double*)PyArray_GETPTR2(c->v_njk,j,k));
              
              double mnkw = *((double*)PyArray_GETPTR2(c->m_nkw,k,wi));
              double vnkw = *((double*)PyArray_GETPTR2(c->v_nkw,k,wi));

              double mnk = *((double*)PyArray_GETPTR1(c->m_nk,k));
              double vnk = *((double*)PyArray_GETPTR1(c->v_nk,k));              

              double newgamma = (alphaval + mnjk) *
                ((betaval + mnkw) / (betasum + mnk)) *
                exp(-1*(vnjk / (2*pow(alphaval + mnjk,2))) -
                    (vnkw / (2*pow(betaval + mnkw,2))) +
                    (vnk / (2*pow(betasum + mnk,2))));
                     
              //printf("betasum+mnk=%f\n",betasum+mnk);
              //printf("alphaval+mnjk=%f\n",alphaval+mnjk);
              //printf("betaval+mnkw=%f\n",betaval+mnkw);
              //printf("betaval=%f\n",betaval);
              //printf("mnkw=%f\n",mnkw);
              //printf("betasum+mnk=%f\n",betasum+mnk);
              //printf("new gamma = %f\n",newgamma);

              // Save new gamma
              *((double*)PyArray_GETPTR2(gamma,k,i)) = newgamma;

              // Update normalization sum
              normsum += newgamma;
            }
          
          // Normalize new gamma, record delta
          cur_delta = 0;
          for(k = 0; k < T; k++) 
            { 
              *((double*)PyArray_GETPTR2(gamma,k,i)) = 
                *((double*)PyArray_GETPTR2(gamma,k,i)) / normsum;
              //printf("new gamma (after re-norm) = %f\n",*((double*)PyArray_GETPTR2(gamma,k,i)));
              cur_delta += 
                fabs(*((double*)PyArray_GETPTR2(gamma,k,i)) - old_gamma[k]);
            }
          // Do we have a new max delta?
          if(cur_delta > max_delta)
            max_delta = cur_delta;
          // Add these gamma back into count mean/var
          count_update(c,ds,i,j,wi,ci);
        }
    }
  free(old_gamma);      
  return max_delta;
}

/**
 * Use variational estimates of count means to estimate theta = P(z|d)
 */
PyArrayObject* est_theta(model_params* mp, dataset* ds, counts* c)
{
  int D = ds->D;
  int T = mp->T;
 
  npy_intp* tdims = malloc(sizeof(npy_intp)*2);
  tdims[0] = D;
  tdims[1] = T;
  PyArrayObject* theta = (PyArrayObject*) 
    PyArray_ZEROS(2,tdims,PyArray_DOUBLE,0);
  free(tdims);
    
  PyArrayObject* jsums = (PyArrayObject*) PyArray_Sum(c->m_njk,
                                                      1,PyArray_DOUBLE,NULL);

  int f=0; // hardwire this for now
  int j,k;
  for(j = 0; j < D; j++) 
    {
      double jsum = *((double*)PyArray_GETPTR1(jsums,j));
      double alphasum = *((double*)PyArray_GETPTR1(mp->alphasum,f)); 
      for(k = 0; k < T; k++)
        {
          double alpha_k = *((double*)PyArray_GETPTR2(mp->alpha,f,k));
          double jkval = *((double*)PyArray_GETPTR2(c->m_njk,j,k));

          // Calc and assign theta entry
          double newval = (alpha_k + jkval) / (alphasum + jsum);
          *((double*)PyArray_GETPTR2(theta,j,k)) = newval;
        }
    }
  return theta;
}

/**
 * Use variational estimates of count means to estimate phi = P(w|z)
 */
PyArrayObject* est_phi(model_params* mp, dataset* ds, counts* c)
{  
  int W = ds->W;
  int T = mp->T;
 
  npy_intp* pdims = malloc(sizeof(npy_intp)*2);
  pdims[0] = T;
  pdims[1] = W;
  PyArrayObject* phi = (PyArrayObject*) 
    PyArray_ZEROS(2,pdims,PyArray_DOUBLE,0);
  free(pdims);

  int k,w;
  for(k = 0; k < T; k++) 
    {
      double ksum = *((double*)PyArray_GETPTR1(c->m_nk,k));
      double betasum = *((double*)PyArray_GETPTR1(mp->betasum,k));
      for(w = 0; w < W; w++) 
        {
          double beta_w = *((double*)PyArray_GETPTR2(mp->beta,k,w));
          double kval = *((double*)PyArray_GETPTR2(c->m_nkw,k,w));
          double newval = (beta_w + kval) / (betasum + ksum);
          *((double*)PyArray_GETPTR2(phi,k,w)) = newval;
        }
    }
  return phi;
}


/**
 * Simultaneously check args and populate structs
 */
static int convert_args(PyObject* docs_w_arg, PyObject* docs_c_arg,
                        PyArrayObject* alpha, PyArrayObject* beta,
                        model_params** p_mp, dataset** p_ds)
  {
    int i;
    int D = PyList_Size(docs_w_arg);
  
    // Get some basic information from parameters
    // (and check dimensionality agreement)
    int T = PyArray_DIM(beta,0);
    int W = PyArray_DIM(beta,1);
    if(T != PyArray_DIM(alpha,1))
      {
        // ERROR
        PyErr_SetString(PyExc_RuntimeError,
                        "Alpha/Beta dimensionality mismatch");
        return ARGS_BAD;
      }

    // Check that all alpha/beta values are non-negative
    //
    double betamin = PyFloat_AsDouble(PyArray_Min(beta,NPY_MAXDIMS,NULL));
    double alphamin = PyFloat_AsDouble(PyArray_Min(alpha,NPY_MAXDIMS,NULL));
    if(betamin <= 0 || alphamin <= 0)       
      {
        // ERROR
        PyErr_SetString(PyExc_RuntimeError,
                        "Negative/zero Alpha or Beta value");
        return ARGS_BAD;
      }

    // Convert documents from PyObject* to int[] 
    //
    int d;
    int* Md = malloc(sizeof(int) * D);
    int** docs_w = malloc(sizeof(int*) * D);
    int** docs_c = malloc(sizeof(int*) * D);
    for(d = 0; d < D; d++)
      {
        PyObject* doc_w = PyList_GetItem(docs_w_arg,d);
        PyObject* doc_c = PyList_GetItem(docs_c_arg,d);
        if(!PyList_Check(doc_w) || !PyList_Check(doc_c))
          {
            // ERROR
            PyErr_SetString(PyExc_RuntimeError,
                            "Non-List element in docs List");
            for(i = 0; i < d; i++)
              {
                free(docs_w[i]);
                free(docs_c[i]);
              }
            free(docs_w);
            free(docs_c);
            free(Md);              
            return ARGS_BAD;
          }
        if(PyList_Size(doc_w) != PyList_Size(doc_c))
          {
            // ERROR
            PyErr_SetString(PyExc_RuntimeError,
                            "doc_w / doc_c length mismatch!");
            for(i = 0; i < d; i++)
              {
                free(docs_w[i]);
                free(docs_c[i]);
              }
            free(docs_w);
            free(docs_c);
            free(Md);              
            return ARGS_BAD;
          }

        Md[d] = PyList_Size(doc_w);
        docs_w[d] = malloc(sizeof(int) * Md[d]);
        docs_c[d] = malloc(sizeof(int) * Md[d]);
        for(i = 0; i < Md[d]; i++)
          {
            // Convert from List elements to int
            docs_w[d][i] = PyInt_AsLong(PyList_GetItem(doc_w,i));
            docs_c[d][i] = PyInt_AsLong(PyList_GetItem(doc_c,i));
            if(docs_w[d][i] < 0 || docs_w[d][i] > (W - 1))
              {
                // ERROR
                PyErr_SetString(PyExc_RuntimeError,
                                "Non-numeric or out of range word");
                for(i = 0; i <= d; i++)
                  {
                    free(docs_w[i]);
                    free(docs_c[i]);
                  }
                free(docs_w);
                free(docs_c);
                free(Md);              
                return ARGS_BAD;
              }
            if(docs_c[d][i] <= 0)
              {
                // ERROR
                PyErr_SetString(PyExc_RuntimeError,
                                "Negative or zero word count");
                for(i = 0; i <= d; i++)
                  {
                    free(docs_w[i]);
                    free(docs_c[i]);
                  }
                free(docs_w);
                free(docs_c);
                free(Md);              
                return ARGS_BAD;
              }
          }
      }

    // Populate dataset struct
    //
    dataset* ds = (dataset*) malloc(sizeof(dataset));
    ds->D = D;
    ds->W = W;
    ds->Md = Md;
    ds->docs_w = docs_w;
    ds->docs_c = docs_c;
 
    // Populate model params struct
    //     
    model_params* mp = (model_params*) malloc(sizeof(model_params));

    mp->alpha = alpha;
    mp->beta = beta;
    mp->T = T;

    mp->alphasum = (PyArrayObject*) PyArray_Sum(alpha,1,PyArray_DOUBLE,NULL);
    mp->betasum = (PyArrayObject*) PyArray_Sum(beta,1,PyArray_DOUBLE,NULL);
    
    *(p_ds) = ds;
    *(p_mp) = mp;
    return ARGS_OK;
  }

/**
 * Initialize gamma using an 'online Gibbs init'
 *
 */
static counts* online_gamma_init(model_params* mp, dataset* ds,
                                 int randseed)
{
  // Init random number generator
  //
  srand((unsigned int) randseed);

  // Get some useful values
  //
  int W = ds->W;
  int D = ds->D;
  int T = mp->T;

  // Convert dataset to 'Gibbs-format'
  // 
  g_dataset* gds = convert_docs(ds);

  // Do the online Gibbs init
  //
  g_counts* gc = gibbs_online_init(mp,gds);

  // Estimate phi and theta from this sample
  //
  PyArrayObject* phi = g_est_phi(mp,gds,gc);
  PyArrayObject* theta = g_est_theta(mp,gds,gc);
  
  // Now we will build up a gamma for each document 
  // using these phi and theta
  //  
  PyArrayObject** gamma = (PyArrayObject**) 
    malloc(sizeof(PyArrayObject*) * ds->D);
  int w,ct; // unique word and its count
  int i,j,k; // [word doc topic] indices

  // For each document
  npy_intp* gdims = (npy_intp*) malloc(sizeof(npy_intp)*2);
  for(j = 0; j < ds->D; j++)
    {
      // Alloc a gamma matrix
      gdims[0] = mp->T;
      gdims[1] = ds->Md[j];
      gamma[j] = (PyArrayObject*) PyArray_ZEROS(2,gdims,PyArray_DOUBLE,0);

      // Populate it for each unique word in doc
      for(i = 0; i < ds->Md[j]; i++)
        {
          w = ds->docs_w[j][i];
          double normsum = 0;
          // Calculate gamma for each topic
          for(k = 0; k < mp->T; k++)
            {
              double newval = (*((double*)PyArray_GETPTR2(theta,j,k))) *
                (*((double*)PyArray_GETPTR2(phi,k,w)));
              (*((double*)PyArray_GETPTR2(gamma[j],k,i))) = newval;
              normsum += newval;
            }
          // Normalize
          for(k = 0; k < mp->T; k++)
            {
              (*((double*)PyArray_GETPTR2(gamma[j],k,i))) = 
                (*((double*)PyArray_GETPTR2(gamma[j],k,i))) / normsum;
            }
        }     
    }
  free(gdims);

  // Save gamma matrices
  //
  ds->gamma = gamma;

  // Alloc and init count matrices and init sample
  //
  counts* c = (counts*) malloc(sizeof(counts));  

  npy_intp* njkdims = malloc(sizeof(npy_intp)*2);
  njkdims[0] = D;
  njkdims[1] = T;
  c->m_njk = (PyArrayObject*) PyArray_ZEROS(2,njkdims,PyArray_DOUBLE,0);
  c->v_njk = (PyArrayObject*) PyArray_ZEROS(2,njkdims,PyArray_DOUBLE,0);
  free(njkdims);

  npy_intp* nkwdims = malloc(sizeof(npy_intp)*2);
  nkwdims[0] = T;
  nkwdims[1] = W;
  c->m_nkw = (PyArrayObject*) PyArray_ZEROS(2,nkwdims,PyArray_DOUBLE,0);
  c->v_nkw = (PyArrayObject*) PyArray_ZEROS(2,nkwdims,PyArray_DOUBLE,0);
  free(nkwdims);

  npy_intp* nkdims = malloc(sizeof(npy_intp));
  nkdims[0] = T;
  c->m_nk = (PyArrayObject*) PyArray_ZEROS(1,nkdims,PyArray_DOUBLE,0);
  c->v_nk = (PyArrayObject*) PyArray_ZEROS(1,nkdims,PyArray_DOUBLE,0);
  free(nkdims);

  // Now populate mean/var matrices from given init gamma
  //
  
  // For each doc
  for(j = 0; j < D; j++)
    {
      // For each unique word in the document
      for(i = 0; i < ds->Md[j]; i++) 
        {
          // Word and its count in this document
          w = ds->docs_w[j][i];
          ct = ds->docs_c[j][i];

          // Update counts for each topic
          count_update(c,ds,i,j,w,ct);
        }
    }  

  // Cleanup
  //
  Py_DECREF(gc->nw);
  Py_DECREF(gc->nd);
  Py_DECREF(gc->nw_colsum);
  free(gc);

  for(j = 0; j < ds->D; j++)
    free(gds->docs[j]);
  free(gds->docs);
  free(gds->doclens);
  free(gds);

  Py_DECREF(phi);
  Py_DECREF(theta);

  // Return resulting counts
  //
  return c;
}

/**
 * Convert SVMLight-style sparse represntation of documents
 * to Gibbs Sampling-style lists of words
 *
 */
static g_dataset* convert_docs(dataset* ds)
{
  int D = ds->D;
  int* doclens = (int*) malloc(sizeof(int) * D);
  int** docs = (int**) malloc(sizeof(int*) * D);

  int d,i,c,ni;
  for(d = 0; d < D; d++)
    {
      // Do a 1st pass over this doc to get length
      int dlen = 0;
      for(i = 0; i < ds->Md[d]; i++)
        {
          dlen += ds->docs_c[d][i];
        }
      // For each unique word, add the corresponding
      // number of copies 
      docs[d] = (int*) malloc(sizeof(int*) * dlen);
      ni = 0; // new index
      for(i = 0; i < ds->Md[d]; i++)
        {
          for(c = 0; c < ds->docs_c[d][i]; c++)
            {
              docs[d][ni] = ds->docs_w[d][i];
              ni++;
            }
        }
      // store resulting document length
      doclens[d] = ni; 
    }

  // package into struct and return
  g_dataset* gds = (g_dataset*) malloc(sizeof(g_dataset));
  gds->D = ds->D;
  gds->W = ds->W;
  gds-> doclens = doclens;
  gds-> docs = docs;
  return gds;
}

/**
 * Do an "online" init of Gibbs chain, adding one word
 * position at a time and then sampling for each new position
 */
static g_counts* gibbs_online_init(model_params* mp, g_dataset* ds)
{   
  // Do some init
  //
  int W = ds->W;
  int D = ds->D;
  int T = mp->T;
 
  // Alloc and init count matrices and init sample
  //
  g_counts* c = (g_counts*) malloc(sizeof(g_counts));  

  npy_intp* nwdims = malloc(sizeof(npy_intp)*2);
  nwdims[0] = W;
  nwdims[1] = T;
  c->nw = (PyArrayObject*) PyArray_ZEROS(2,nwdims,PyArray_INT,0);
  free(nwdims);

  npy_intp* nddims = malloc(sizeof(npy_intp)*2);
  nddims[0] = D;
  nddims[1] = T;
  c->nd =  (PyArrayObject*) PyArray_ZEROS(2,nddims,PyArray_INT,0);
  c->nw_colsum = (PyArrayObject*) PyArray_Sum(c->nw,0,PyArray_INT,NULL);
  free(nddims);
  
  // Build init z sample, one word at a time
  //

  // Temporary array used for sampling
  double* num = (double*) malloc(sizeof(double)*T);

  // For each doc in corpus
  int d,i,j;
  for(d = 0; d < D; d++) 
    {
      // Get this doc and f-label
      int* doc = ds->docs[d];
      int doclen = ds->doclens[d];
      int f = 0;

      // For each word in doc
      for(i = 0; i < doclen; i++)
        {      
          int w_i = doc[i];
	
          // For each topic, calculate numerators
          double norm_sum = 0;
          for(j = 0; j < T; j++) 
            { 
              double alpha_j = *((double*)PyArray_GETPTR2(mp->alpha,f,j));
              double beta_i = *((double*)PyArray_GETPTR2(mp->beta,j,w_i));
              double betasum = *((double*)PyArray_GETPTR1(mp->betasum,j));	
              double denom_1 = *((int*)PyArray_GETPTR1(c->nw_colsum,j)) + betasum;

              // Calculate numerator for this topic
              // (NOTE: alpha denom omitted, since same for all topics)
              num[j] = ((*((int*)PyArray_GETPTR2(c->nw,w_i,j)))+beta_i) / denom_1;
              num[j] = num[j] * (*((int*)PyArray_GETPTR2(c->nd,d,j))+alpha_j);

              norm_sum += num[j];
            }
	
          // Draw a sample
          //   
          j = mult_sample(num,norm_sum);
	
          // Update count/cache matrices
          //
          (*((int*)PyArray_GETPTR2(c->nw,w_i,j)))++;
          (*((int*)PyArray_GETPTR2(c->nd,d,j)))++;
          (*((int*)PyArray_GETPTR1(c->nw_colsum,j)))++;
        }
    }
  // Cleanup, put all counts in struct, and return
  //
  free(num);
  return c;
}

/**
 * Use final sample to estimate theta = P(z|d)
 */
PyArrayObject* g_est_theta(model_params* mp, g_dataset* ds, g_counts* c)
{
  int D = ds->D;
  int T = mp->T;
 
  npy_intp* tdims = malloc(sizeof(npy_intp)*2);
  tdims[0] = D;
  tdims[1] = T;
  PyArrayObject* theta = (PyArrayObject*) 
    PyArray_ZEROS(2,tdims,PyArray_DOUBLE,0);
  free(tdims);
    
  PyArrayObject* rowsums = (PyArrayObject*) PyArray_Sum(c->nd,1,PyArray_DOUBLE,NULL);

  int d,t;
  for(d = 0; d < D; d++) 
    {
      double rowsum = *((double*)PyArray_GETPTR1(rowsums,d));
      int f = 0;
      double alphasum = *((double*)PyArray_GETPTR1(mp->alphasum,f));
      for(t = 0; t < T; t++)
        {
          double alpha_t = *((double*)PyArray_GETPTR2(mp->alpha,f,t));
          int ndct = *((int*)PyArray_GETPTR2(c->nd,d,t));

          // Calc and assign theta entry
          double newval = (ndct + alpha_t) / (rowsum+alphasum);
          *((double*)PyArray_GETPTR2(theta,d,t)) = newval;
        }
    }
  return theta;
}

/**
 * Use final sample to estimate phi = P(w|z)
 */
PyArrayObject* g_est_phi(model_params* mp, g_dataset* ds, g_counts* c)
{  
  int W = ds->W;
  int T = mp->T;
 
  npy_intp* pdims = malloc(sizeof(npy_intp)*2);
  pdims[0] = T;
  pdims[1] = W;
  PyArrayObject* phi = (PyArrayObject*) 
    PyArray_ZEROS(2,pdims,PyArray_DOUBLE,0);
  free(pdims);

  int t,w;
  for(t = 0; t < T; t++) 
    {
      int colsum = (*((int*)PyArray_GETPTR1(c->nw_colsum,t)));
      double betasum = *((double*)PyArray_GETPTR1(mp->betasum,t));
      for(w = 0; w < W; w++) 
        {
          double beta_w = *((double*)PyArray_GETPTR2(mp->beta,t,w));
          int nwct = *((int*)PyArray_GETPTR2(c->nw,w,t));
          double newval = (beta_w + nwct) / (betasum + colsum);
          *((double*)PyArray_GETPTR2(phi,t,w)) = newval;
        }
    }
  return phi;
}

/**
 * Draw a multinomial sample propto vals
 * 
 * (!!! we're assuming sum is the correct sum for vals !!!)
 * 
 */
static int mult_sample(double* vals, double norm_sum)
{
  double rand_sample = unif() * norm_sum;
  double tmp_sum = 0;
  int j = 0;
  while(tmp_sum < rand_sample || j == 0) {
    tmp_sum += vals[j];
    j++;
  }
  return j - 1;
}

//
// PYTHON EXTENSION BOILERPLATE BELOW
//

// Defines the module method table
PyMethodDef methods[] = 
  {
    {"cvbLDA", (PyCFunction) cvbLDA, 
     METH_VARARGS | METH_KEYWORDS, "Run CvbLDA"},
    {NULL, NULL, 0, NULL}  // Is a 'sentinel' (?)
  };

// This is a macro that does stuff for us (linkage, declaration, etc)
PyMODINIT_FUNC 
initcvbLDA() // Passes method table to init our module
{
  (void) Py_InitModule("cvbLDA", methods); 
  import_array(); // Must do this to satisfy NumPy (!)
}
