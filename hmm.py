from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict
        
        

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for s in range(S):
            
            alpha[s,0] = self.pi[s] * self.B[s,self.obs_dict[Osequence[0]]]
        for t in range(1,L):
            for j in range(S):
                sig = 0
                for h in range(S):
                    sig = sig + alpha[h,t-1]*self.A[h,j]
                alpha[j,t] = self.B[j,self.obs_dict[Osequence[t]]] * sig
        
        self.alpha = alpha
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        
        for s in range(S):
            beta[s,L-1] = 1
        for t in range(L-2,-1,-1):
            for j in range(S):
                sigma = 0
                for h in range(S):
                    sigma = sigma + beta[h,t+1]*self.A[j,h]*self.B[h,self.obs_dict[Osequence[t+1]]]
                beta[j,t] =  sigma
                
        self.beta = beta
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        sigma = 0
        S = len(self.pi)
        L = len(Osequence)
        for h in range(S):
            sigma = sigma + self.alpha[h,L-1]
        prob = sigma
        
        self.PO1T = prob
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
        GAMMA
        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        if self.PO1T == 0:
            return prob
        prob = self.alpha*self.beta/self.PO1T
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L
        SI
        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        if self.PO1T == 0:
            return prob
        ###################################################
        # Edit here
        ###################################################
        for t in range(0,L-1):
            for s in range(S):
                for sp in range(S):
                    prob[s,sp,t] = self.alpha[s,t]*self.A[s,sp]*self.B[sp,self.obs_dict[Osequence[t+1]]]*self.beta[sp,t+1]/self.PO1T
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        
        ###################################################
        # Q3.3 Edit here
        ###################################################
        ind_state = dict([(value, key) for key, value in self.state_dict.items()]) 
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S,L])
        dp = np.zeros([S,L])
        pathid = np.zeros(L)
        for s in range(S):
            delta[s,0] = self.pi[s]*self.B[s,self.obs_dict[Osequence[0]]]
        for t in range(1,L):
            for s in  range(S):
                Adelta = self.A[...,s]*delta[...,t-1]
                delta[s,t] = self.B[s,self.obs_dict[Osequence[t]]]*max(Adelta)
                dp[s,t] = np.argmax(Adelta)
    
        pathid[L-1] = np.argmax(delta[...,L-1])
        for t in range(L-2,-1,-1):
            pathid[t] = dp[int(pathid[t+1]),t+1]
        for i in range(len(pathid)):
            path.append(ind_state[pathid[i]] )      
        return path
