import numpy
import os
import math
from scipy.misc import logsumexp

class _BaseHMM(object):
	def __init__(self,n,m,precision=numpy.double,verbose=False):
		self.n = n
		self.m = m

		self.precision = precision
		self.verbose = verbose

		self._eta = self._eta1


	def _eta1(self,t,T):
		'''
		Governs how each sample in the time series should be weighed.
		This is the default case where each sample has the same weigh, 
		i.e: this is a 'normal' HMM.
		'''
		return 1.

	def train(self, observations, iterations=1,epsilon=0.0001,thres=-0.001):
		'''
		Updates the HMMs parameters given a new set of observed sequences.

		observations can either be a single (1D) array of observed symbols, or when using
		a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
		time sample (multiple features).

		Training is repeated 'iterations' times, or until log likelihood of the model
		increases by less than 'epsilon'.

		'thres' denotes the algorithms sensitivity to the log likelihood decreasing
		from one iteration to the other.
		'''        
		self._mapB(observations)

		for i in xrange(iterations):
		    prob_old, prob_new = self.trainiter(observations, i)
		    
		    # if (self.verbose):      
		    #     print "iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new, ", converging =", ( prob_new-prob_old > thres )
		        
		    if ( abs(prob_new-prob_old) < epsilon ):
		        # converged
		        break

	def trainiter(self,observations, i):
		'''
		A single iteration of an EM algorithm, which given the current HMM,
		computes new model parameters and internally replaces the old model
		with the new one.

		Returns the log likelihood of the old model (before the update),
		and the one for the new model.
		'''        
		# call the EM algorithm
		new_model = self._baumwelch(observations)

		# calculate the log likelihood of the previous model
		prob_old = self.forward(observations, cache=True)

		# update the model with the new estimation
		self._updatemodel(new_model, i)

		# calculate the log likelihood of the new model. Cache set to false in order to recompute probabilities of the observations give the model.
		prob_new = self.forward(observations, cache=False)

		return prob_old, prob_new

	def _baumwelch(self,observations):
		'''
		An EM(expectation-modification) algorithm devised by Baum-Welch. Finds a local maximum
		that outputs the model that produces the highest probability, given a set of observations.

		Returns the new maximized model parameters
		'''        
		# E step - calculate statistics
		stats = self._calcstats(observations)

		# M step
		return self._reestimate(stats,observations)

	def _calcstats(self,observations):
		'''
		Calculates required statistics of the current model, as part
		of the Baum-Welch 'E' step.

		Deriving classes should override (extend) this method to include
		any additional computations their model requires.

		Returns 'stat's, a dictionary containing required statistics.
		'''
		stats = {}

		stats['alpha'] = self._calcalpha(observations)
		stats['beta'] = self._calcbeta(observations)
		stats['xi'] = self._calcxi(observations,stats['alpha'],stats['beta'])
		stats['gamma'] = self._calcgamma(stats['xi'],len(observations))

		return stats

	def _calcalpha(self,observations):
		'''
		Calculates 'alpha' the forward variable. alpha is log probability

		The alpha variable is a numpy array indexed by time, then state (TxN).
		alpha[t][i] = the probability of being in state 'i' after observing the 
		first t symbols.
		'''        
		alpha = numpy.zeros((len(observations),self.n),dtype=self.precision)

		# init stage - alpha_1(x) = pi(x)b_x(O1)
		# print '=======pi========='
		# print self.pi
		for x in xrange(self.n):
		    alpha[0][x] = self.pi[x] + self.B_map[x][0]

		# induction
		for t in xrange(1,len(observations)):
		    for j in xrange(self.n):
		    	temp = []
		        for i in xrange(self.n):
		            temp.append(alpha[t-1][i] + self.A[i][j])
		        alpha[t][j] = logsumexp(temp)
		        alpha[t][j] += self.B_map[j][t]
    
		return alpha

	def _calcbeta(self,observations):
		'''
		Calculates 'beta' the backward variable.

		The beta variable is a numpy array indexed by time, then state (TxN).
		beta[t][i] = the probability of being in state 'i' and then observing the
		symbols from t+1 to the end (T).
		'''        
		beta = numpy.zeros((len(observations),self.n),dtype=self.precision)

		# init stage
		for s in xrange(self.n):
		    beta[len(observations)-1][s] = 0.

		# induction
		for t in xrange(len(observations)-2,-1,-1):
		    for i in xrange(self.n):
		    	temp = []
		        for j in xrange(self.n):
		            temp.append(self.A[i][j] + self.B_map[j][t+1] + beta[t+1][j])
		        beta[t][i] = logsumexp(temp)    
		return beta

	def _calcxi(self,observations,alpha=None,beta=None):
		'''
		Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

		The xi variable is a numpy array indexed by time, state, and state (TxNxN).
		xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
		time 't+1' given the entire observation sequence.
		'''        
		if alpha is None:
		    alpha = self._calcalpha(observations)
		if beta is None:
		    beta = self._calcbeta(observations)
		xi = numpy.zeros((len(observations),self.n,self.n),dtype=self.precision)


		for t in xrange(len(observations) - 1):
		    denom = []
		    for i in xrange(self.n):
		        for j in xrange(self.n):
		            thing = 0.0
		            thing += alpha[t][i]
		            thing += self.A[i][j]
		            thing += self.B_map[j][t+1]
		            thing += beta[t+1][j]
		            denom.append(thing)
		    denom = logsumexp(denom)        
		    # print denom
		    for i in xrange(self.n):
		        for j in xrange(self.n):
		            numer = 0.0
		            numer += alpha[t][i]
		            numer += self.A[i][j]
		            numer += self.B_map[j][t+1]
		            numer += beta[t+1][j]
		            xi[t][i][j] = numer - denom
		#             if denom == 0:
		#             	xi[t][i][j] = 0
		#             else:
		#             	xi[t][i][j] = numer/denom
		# print '========xi========'
		# print 'alpha'
		# print alpha[len(observations) - 2]
		# print 'beta'
		# print beta[len(observations) - 2]
		# print 'B_map'
		# print self.B_map[0][len(observations) - 1]
		# print '===xi==='
		# print xi
		# print xi[len(observations) - 1]
		return xi

	def _calcgamma(self,xi,seqlen):
		'''
		Calculates 'gamma' from xi.

		Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
		in state 'i' at time 't' given the full observation sequence.
		'''        
		gamma = numpy.zeros((seqlen,self.n),dtype=self.precision)

		for t in xrange(seqlen):
		    for i in xrange(self.n):
	        	gamma[t][i] = logsumexp(xi[t][i])

		# print '========gamma========'
		# print gamma
		return gamma

	def _reestimate(self,stats,observations):
		'''
		Performs the 'M' step of the Baum-Welch algorithm.

		Deriving classes should override (extend) this method to include
		any additional computations their model requires.

		Returns 'new_model', a dictionary containing the new maximized
		model's parameters.
		'''        
		new_model = {}

		# new init vector is set to the frequency of being in each step at t=0 
		new_model['pi'] = stats['gamma'][0]
		new_model['A'] = self._reestimateA(observations,stats['xi'],stats['gamma'])

		return new_model

	def _reestimateA(self,observations,xi,gamma):
		'''
		Reestimation of the transition matrix (part of the 'M' step of Baum-Welch).
		Computes A_new = expected_transitions(i->j)/expected_transitions(i)

		Returns A_new, the modified transition matrix. 
		'''
		A_new = numpy.zeros((self.n,self.n),dtype=self.precision)
		for i in xrange(self.n):
		    for j in xrange(self.n):
		        numer = []
		        denom = []
		        for t in xrange(len(observations)-1):
		            numer.append(self._eta(t,len(observations)-1)*xi[t][i][j])
		            denom.append(self._eta(t,len(observations)-1)*gamma[t][i])
		        numer = logsumexp(numer)
		        denom = logsumexp(denom)
		        # if denom == 0:
		        # 	A_new[i][j] = 0
		        # else:
		        # 	A_new[i][j] = numer/denom
		        A_new[i][j] = numer - denom
		        # print 'numer ' + str(numer) + ' denom ' + str(denom)
		# print 'A_new=============='
		# print A_new
		return A_new

	def _updatemodel(self,new_model, i):
		'''
		Replaces the current model parameters with the new ones.
		'''
		self.pi = new_model['pi']
		self.A = new_model['A']

	def forward(self,observations, cache=False):
		'''
		Forward procedure is used to efficiently calculate the probability of the observation, given the model - P(O|model)
		alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the observation up to time t, given the model.

		The returned value is the log of the probability, i.e: the log likehood model, give the observation - logL(model|O).

		In the discrete case, the value returned should be negative, since we are taking the log of actual (discrete)
		probabilities. In the continuous case, we are using PDFs which aren't normalized into actual probabilities,
		so the value could be positive.
		'''
		if (cache==False):
		    self._mapB(observations)

		alpha = self._calcalpha(observations)
		sum_alpha = logsumexp(alpha[-1])
		# if sum_alpha == 0:
		# 	return 0
		# else:
		# 	return numpy.log(sum_alpha)
		return sum_alpha

	def _saveModel(self, path):
		'''
		Save model parameters to file
		'''
		raise NotImplementedError("a mapping function for _saveModel must be implemented")

	def _mapB(self,observations):
		'''
		Deriving classes should implement this method, so that it maps the observations'
		mass/density Bj(Ot) to Bj(t).

		This method has no explicit return value, but it expects that 'self.B_map' is internally computed
		as mentioned above. 'self.B_map' is an (TxN) numpy array.

		The purpose of this method is to create a common parameter that will conform both to the discrete
		case where PMFs are used, and the continuous case where PDFs are used.

		For the continuous case, since PDFs of vectors could be computationally 
		expensive (Matrix multiplications), this method also serves as a caching mechanism to significantly
		increase performance.
		'''
		raise NotImplementedError("a mapping function for B(observable probabilities) must be implemented")

	def decode(self, observations):
		'''
		Find the best state sequence (path), given the model and an observation. i.e: max(P(Q|O,model)).

		This method is usually used to predict the next state after training. 
		'''        
		# use Viterbi's algorithm. It is possible to add additional algorithms in the future.
		return self._viterbi(observations)
    
	def _viterbi(self, observations):
		'''
		Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
		very similar to the forward-backward algorithm, with the added step of maximization and eventual
		backtracing.

		delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
		that generates the highest probability.

		psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1), 
		i.e: the previous state.
		'''
		# similar to the forward algorithm, we need to make sure that we're using fresh data for the given observations.
		self._mapB(observations)

		delta = numpy.zeros((len(observations),self.n),dtype=self.precision)
		psi = numpy.zeros((len(observations),self.n),dtype=self.precision)

		# init
		for x in xrange(self.n):
		    delta[0][x] = numpy.exp(self.pi[x])*numpy.exp(self.B_map[x][0])
		    psi[0][x] = 0

		# induction
		for t in xrange(1,len(observations)):
		    for j in xrange(self.n):
		        for i in xrange(self.n):
		            if (delta[t][j] < delta[t-1][i]*numpy.exp(self.A[i][j])):
		                delta[t][j] = delta[t-1][i]*numpy.exp(self.A[i][j])
		                psi[t][j] = i
		        delta[t][j] *= numpy.exp(self.B_map[j][t])

		# termination: find the maximum probability for the entire sequence (=highest prob path)
		p_max = 0 # max value in time T (max)
		path = numpy.zeros((len(observations)),dtype=self.precision)
		for i in xrange(self.n):
		    if (p_max < delta[len(observations)-1][i]):
		        p_max = delta[len(observations)-1][i]
		        path[len(observations)-1] = i

		# path backtracing
		#        path = numpy.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
		for i in xrange(1, len(observations)):
		    path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]

		path = [int(p+2) for p in path]
		return path



