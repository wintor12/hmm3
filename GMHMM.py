from _ContinuousHMM import _ContinuousHMM
import numpy
from scipy.stats import multivariate_normal
import os
import sys

class GMHMM(_ContinuousHMM):

    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        See _ContinuousHMM constructor for more information
        '''
        _ContinuousHMM.__init__(self,n,m,d,A,means,covars,w,pi,min_std,init_type,precision,verbose)

    def _pdf(self,x,mean,covar):
        '''
        Gaussian PDF function
        '''  

        var = multivariate_normal(mean=mean, cov=covar)
        pdfval = var.pdf(x)
        if pdfval == 0:
            pdfval = 1.97121603e-099
        #     print 'wrong============'
        #     print 'mean'
        #     print mean
        #     print 'covar'
        #     print covar
        #     print 'x'
        #     print x
        #     sys.exit()

        return pdfval