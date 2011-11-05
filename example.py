import pdb

from numpy import *
from cvbLDA import cvbLDA

# alpha and beta *must* be NumPy Array objects
#
# Their dimensionalities implicitly specify:
# -the number of topics T
# -the vocabulary size W
# -number of f-label values F (just 1 for 'standard' LDA)
#
# alpha = F x T
# beta = T x W
#
(T,W) = (3,5)
alpha = .1 * ones((1,T))
beta = .1 * ones((T,W))

# docs_w = List of Lists (unique word indices)
# docs_c = List of Lists (unique word counts)
#
# (this is similar to sparse SVMLight-style input)
#
docs_w = [[1,2],
          [1,2],
          [3,4],
          [3,4],
          [0],
          [0]]
docs_c = [[2,1],
          [4,1],
          [3,1],
          [4,2],
          [5],
          [4]]

# Stopping conditions for inference:
# -stop after maxiter iterations
# -stop once sum of absolute changes in all gamma variational
#  parameters for a single iteration falls below convtol
# (whichever occurs FIRST)
# 
# If these parameters are not supplied, stop after 100 iterations
#
(maxiter,convtol) = (10,.01)

# Do CVB inference for LDA
#
(phi,theta,gamma) = cvbLDA(docs_w,docs_c,alpha,beta,
                           maxiter=maxiter,verbose=1,convtol=convtol)

# theta is the matrix of document-topic probabilities
# (estimated from expected counts under variational posterior)
# 
# theta = D x T
# theta[di,zj] = P(z=zj | d=di)
#
print ''
print 'Theta - P(z|d)'
print str(theta)
print ''

# phi is the matrix of topic-word probabilities 
# (estimated from expected counts under variational posterior)
# 
# phi = T x W
# phi[zj,wi] = P(w=wi | z=zj)
#
print ''
print 'Phi - P(w|z)'
print str(phi)
print ''

# Since the simple documents we created and fed into cvbLDA exhibit such
# clearly divided word usage patterns, the resulting phi and theta
# should reflect these patterns nicely

#
# These are the final variational parameters
#
for (di,g) in enumerate(gamma):
    print 'Document %d gamma' % di
    print str(g)
    print ''
