import unittest
import pdb

from numpy import *
from cvbLDA import cvbLDA

# If deltaLDA extension module present,
# we will sanity check against it
hasDelta = True
try:
    from deltaLDA import deltaLDA
except:
    hasDelta = False

class TestCvbLDA(unittest.TestCase):
    
    def setUp(self):
        """ Set up base data/parameter values """
        (self.T,self.W) = (3,5)
        self.alpha = .1 * ones((1,self.T))
        self.beta = ones((self.T,self.W))
        self.docs_w = [[1,2],
                       [1,2],
                       [3,4],
                       [3,4],
                       [0],
                       [0]]
        self.docs_c = [[2,1],
                       [4,1],
                       [3,1],
                       [4,2],
                       [5],
                       [4]]
        (self.maxiter,self.convtol) = (50,.001)
        # equality tolerance for testing
        self.tol = 1e-6


    def matAgree(self,mat1,mat2):
        """
        Given 2 NumPy matrices,
        check that all values agree within self.tol         
        """    
        return (abs(mat1-mat2)).sum(axis=None) < self.tol


    def matProb(self,mat):
        """
        Given a NumPy matrix,
        check that all values >= 0, sum to 1 (valid prob dist)
        """
        sumto1 = all([abs(val - float(1)) < self.tol
                      for val in mat.sum(axis=1)])
        geq0 = all([val >= 0 for val in
                    mat.reshape(mat.size)])
        return sumto1 and geq0


    #
    # These are to test *correct* operation
    #    

    def testStandard(self):
        """ Test standard LDA with base data/params """
        (phi,theta,gamma) = cvbLDA(self.docs_w,self.docs_c,
                                   self.alpha,self.beta,
                                   maxiter=self.maxiter,
                                   convtol=self.convtol)

        # theta should clust docs [0,1], [2,3], [4,5]
        maxtheta = argmax(theta,axis=1)
        self.assert_(maxtheta[0] == maxtheta[1])
        self.assert_(maxtheta[2] == maxtheta[3])
        self.assert_(maxtheta[4] == maxtheta[5])
        # theta valid prob matrix
        self.assert_(self.matProb(theta))

        # corresponding phi should emph [1,2], [3,4], [0]
        maxphi = argmax(phi,axis=1)
        self.assert_(maxphi[maxtheta[0]] == 1)
        self.assert_(maxphi[maxtheta[2]] == 3)
        self.assert_(maxphi[maxtheta[4]] == 0)
        # phi valid prob matrix
        self.assert_(self.matProb(phi))

    def testSanity(self):
        """
        Sanity check online Gibbs init scheme against deltaLDA
        """
        # Don't even try unless deltaLDA module present
        if(not hasDelta):
            return
        
        randseed = 194582

        # Use 1 'online' Gibbs sample to build initial gamma
        gibbs_docs = [[1,1,2],
                      [1,1,1,1,2],
                      [3,3,3,4],
                      [3,3,3,3,4,4],
                      [0,0,0,0,0],
                      [0,0,0,0]]
        numsamp = 0
        (phi,theta,sample) = deltaLDA(gibbs_docs,self.alpha,self.beta,
                                      numsamp,randseed)
        gamma_init = []
        for (d,di) in zip(self.docs_w,range(len(self.docs_w))):
            gamma = zeros((self.T,len(d)))
            for (w,i) in zip(d,range(len(d))):
                gamma[:,i] = theta[di,:] * phi[:,w]
                # normalize
                gamma[:,i] = gamma[:,i] / gamma[:,i].sum()
            # save
            gamma_init.append(gamma)        
        
        # Run cvbLDA with this gamma
        (gphi,gtheta,gamma) = cvbLDA(self.docs_w,self.docs_c,
                                     self.alpha,self.beta,
                                     gamma_init=gamma_init,
                                     maxiter=self.maxiter,
                                     convtol=self.convtol)

        # Run cvbLDA no init gamma, same randseed
        (phi,theta,gamma) = cvbLDA(self.docs_w,self.docs_c,
                                   self.alpha,self.beta,
                                   randseed=randseed,
                                   maxiter=self.maxiter,
                                   convtol=self.convtol)

        self.assert_(self.matAgree(phi,gphi))
        self.assert_(self.matProb(phi))
        self.assert_(self.matProb(gphi))
        
        self.assert_(self.matAgree(theta,gtheta))
        self.assert_(self.matProb(theta))
        self.assert_(self.matProb(gtheta))

    #
    # These test that the program fails correctly on *bad input*
    #    

    """
    Test bad document data
    """
    
    def testNonListDoc(self):
        """  Non-list doc """
        self.docs_w[0] = None
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,
                          maxiter=self.maxiter,convtol=self.convtol)
    def testNegWord(self):
        """  Bad word (negative) """
        self.docs_w[0][-1] = -1
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,       
                          maxiter=self.maxiter,convtol=self.convtol)
    def testBigWord(self):
        """  Bad word (too big)                 """
        self.docs_w[0][-1] = 5
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,       
                          maxiter=self.maxiter,convtol=self.convtol)
    def testNonNumWord(self):
        """  Bad word (non-numeric) """
        self.docs_w[0][-1] = ''
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,       
                          maxiter=self.maxiter,convtol=self.convtol)

    """
    Test bad alpha/beta values
    """

    def testNegAlpha(self):
        """  Negative alpha """
        self.alpha[0,1] = -1
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,
                          maxiter=self.maxiter,convtol=self.convtol)
    def testNegBeta(self):
        """  Negative beta """
        self.beta[1,2] = -1
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,
                          maxiter=self.maxiter,convtol=self.convtol)
    def testAlphaBetaDim(self):
        """  Alpha/Beta dim mismatch """
        self.alpha = .1 * ones((1,4))
        self.beta = ones((3,5))
        self.assertRaises(RuntimeError,cvbLDA,self.docs_w,self.docs_c,
                          self.alpha,
                          self.beta,
                          maxiter=self.maxiter,convtol=self.convtol)

    
"""  Run the unit tests! """
if __name__ == '__main__':
    unittest.main()
