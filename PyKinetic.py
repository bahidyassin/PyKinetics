import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import time
import random
import statistics
from numpy import linalg as LA
import matplotlib.animation as animation
import copy
import scipy as sp
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import odeint
import matplotlib 
from numba import jit
import torch
import seaborn as sns
import os
import imageio
import glob
import shutil
def BasisIntegralGenerator(n,a=0,b=1):
    x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
    SetXpoints = set(x_points)
    w = torch.tensor(barycentric_weights(x_points))
    qn = 100 # n #int(n/2)
    gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
    gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

    LowerBound, UpperBound = 0, 1

    transformed_nodes = 0.5 * (UpperBound - LowerBound) * gaussnodes + 0.5 *  (UpperBound + LowerBound)
    transformed_weights = 0.5 *  (UpperBound - LowerBound) * guassweights

    Numerator = w[None, :]/(transformed_nodes[:, None] - x_points[None, :])
    
    Denominator = torch.sum(Numerator, dim = 1 )
    # print(Numerator.shape, Denominator.shape, transformed_weights.shape)
    Bp = transformed_weights[:,None]*Numerator/Denominator[:,None]

    Result = torch.sum(Bp, dim = 0)
    return Result

def ZeroFunc(*args):
    return 0
def OneFunc(*args):
    return 1
def ChebyshevNodes(n, a=0, b=1):

    # Calculate Chebyshev nodes in the interval [-1, 1]
    chebyshev_nodes = np.cos((2 * np.arange(n) + 1) / (2 * n) * np.pi)

    # Transform nodes to the interval [a, b]
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * chebyshev_nodes
    nodes.sort()
    return np.array(list(nodes))
def barycentric_weights(x_points):
    n = len(x_points)
    w = np.ones(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (x_points[i] - x_points[j])
    
    return w

def barycentric_polynomial(x, x_points, i,w):
    # w = barycentric_weights(x_points)  # Compute the barycentric weights
    if x<0 or x>1:
        return 0
    n = len(x_points)
    
    numerator = w[i] / (x - x_points[i]) if x != x_points[i] else 1.0  # Handle division by zero
    denominator = 0.0
    
    for j in range(n):
        if x != x_points[j]:
            denominator += w[j] / (x - x_points[j])
        else:
            # If x matches a node exactly, the polynomial evaluates to 1 at that node and 0 at others.
            return 1.0 if i == j else 0.0
    
    return numerator / denominator
def lagrange_basis_at_x(x, nodes, w):
    diffs = x - nodes
    idx = np.where(np.abs(diffs) < 1e-14)[0]
    if idx.size > 0:
        j = idx[0]
        ell = np.zeros_like(nodes)
        ell[j] = 1.0
        return ell
    numer = w / diffs
    denom = numer.sum()
    return numer / denom
###################################################################################################################################################################
## Transitions
#################################################################################################################################################################

##### Right No Encounter
class TransitionRight():
    def __init__(self, gamma, rt):
        self.TransitionType = 'NoEncounter'
        self.gamma = gamma
        self.rate = rt
        
    def MassAddition(self, n, a = 0, b = 1):
        nodes = ChebyshevNodes(n)
        w = barycentric_weights(nodes)
        M = np.zeros((n, n))
        for i, ui in enumerate(nodes):
            xstar = (ui - self.gamma) / (1.0 - self.gamma)
            if 0.0 <= xstar <= 1.0:
                ell = lagrange_basis_at_x(xstar, nodes, w)
                M[i, :] = (1.0 / (1.0 - self.gamma)) * ell
            else:
                M[i, :] = 0.0
        return torch.from_numpy(self.rate*M)
    def MassSubstraction(self, n, a = 0, b = 1):
        
        return self.rate*torch.ones(n)
##### Left No Encounter
class TransitionLeft():
    def __init__(self, gamma, rt):
        self.TransitionType = 'NoEncounter'
        self.gamma = gamma
        self.rate = rt
        
    def MassAddition(self, n, a = 0, b = 1):
        nodes = ChebyshevNodes(n)
        w = barycentric_weights(nodes)
        M = np.zeros((n, n))
        for i, ui in enumerate(nodes):
            xstar = (ui ) / (1.0 - self.gamma)
            if 0.0 <= xstar <= 1.0:
                ell = lagrange_basis_at_x(xstar, nodes, w)
                M[i, :] = (1.0 / (1.0 - self.gamma)) * ell
            else:
                M[i, :] = 0.0
        return torch.from_numpy(self.rate*M)
    def MassSubstraction(self, n, a = 0, b = 1):
        
        return self.rate*torch.ones(n)

##### Chi no Encounter
class TransitionChi2Chi():
    def __init__(self, loboundx, upboundx, loboundu, upboundu, rt):
        self.TransitionType = 'NoEncounterChi'
        self.loboundx = loboundx
        self.upboundx = upboundx
        self.loboundu = loboundu
        self.upboundu = upboundu
        self.rate = rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
        pLowerBound, pUpperBound =  torch.zeros(len(x_points)) + self.loboundx , torch.zeros(len(x_points)) + self.upboundx
        
        # return pLowerBound, pUpperBound
        
        ptransformed_nodes = 0.5 * (pUpperBound[None,:] - pLowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (pUpperBound[None,:] + pLowerBound[None,:])
        ptransformed_weights = 0.5 *  (pUpperBound[None,:] - pLowerBound[None,:]) * guassweights[:, None]
       
         

        pNumerator = w[None, None, :]/(ptransformed_nodes[:, :, None] - x_points[None, None, :])
        
        

        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        
        
        Bp = pNumerator/pDenominator[:,:,None]

        Integrandp =  ptransformed_weights[:,:, None]*(Bp[:,:,])
        
        Result = torch.sum(Integrandp, dim = 0)[:,:]
        chiu = torch.ones(len(x_points))
        chiu[x_points < self.loboundu] = 0
        chiu[x_points > self.upboundu] = 0
        print(chiu[:, None], Result)
        return (1/(self.upboundu - self.loboundu))*self.rate*chiu[:,None]*Result

    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        
        chiu = torch.ones(len(x_points))
        chiu[x_points < self.loboundx] = 0
        chiu[x_points > self.upboundx] = 0
        return self.rate*chiu*torch.ones(n)
##### Chi with Encounter
class TransitionChiXY():
    def __init__(self, lobound, upbound, rt):
        self.TransitionType = 'EncounterChi'
        self.lobound = lobound
        self.upbound = upbound
        self.rate = rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
        pLowerBound, pUpperBound =  torch.zeros(len(x_points)) + self.lobound , torch.zeros(len(x_points)) + self.upbound
        qLowerBound, qUpperBound =  torch.zeros(len(x_points)), torch.ones(len(x_points))
        # return pLowerBound, pUpperBound
        
        ptransformed_nodes = 0.5 * (pUpperBound[None,:] - pLowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (pUpperBound[None,:] + pLowerBound[None,:])
        ptransformed_weights = 0.5 *  (pUpperBound[None,:] - pLowerBound[None,:]) * guassweights[:, None]
        qtransformed_nodes = 0.5 * (qUpperBound[None,:] - qLowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (qUpperBound[None,:] + qLowerBound[None,:])
        qtransformed_weights = 0.5 *  (qUpperBound[None,:] - qLowerBound[None,:]) * guassweights[:, None]
         

        pNumerator = w[None, None, :]/(ptransformed_nodes[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qtransformed_nodes[:, :, None] - x_points[None, None, :])


        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]

        Integrandp =  ptransformed_weights[:,:, None, None]*(Bp[:,:,:,None])
        
        Bq = qNumerator/qDenominator[:,:,None]
        
        Integrandq =  qtransformed_weights[:,:, None, None]*(Bq[:,:,None,:])
        Result = torch.sum(Integrandp, dim = 0)*torch.sum(Integrandq, dim = 0)
        return self.rate*Result

    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
    
        chibounds = torch.ones(len(x_points))
        chibounds[x_points < self.lobound] = 0
        chibounds[x_points > self.upbound] = 0
        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))), torch.tensor(np.ones(len(x_points)))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        Numerator = w[None, None, :]/(transformed_nodes[:, :, None] - x_points[None, None, :])
    
        Denominator = torch.sum(Numerator, dim = 2 )
    
        Bp = transformed_weights[:,:, None]*Numerator/Denominator[:,:,None]
        # return Bp.shape
        Result = chibounds[:, None] * torch.sum(Bp, dim = 0)
        return self.rate*Result
##### Double Chi
class TransitionChi2ChiXY():
    def __init__(self, loboundx, upboundx, loboundu, upboundu, rt):
        self.TransitionType = 'EncounterChi'
        self.loboundx = loboundx
        self.upboundx = upboundx
        self.loboundu = loboundu
        self.upboundu = upboundu
        self.rate = rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
        pLowerBound, pUpperBound =  torch.zeros(len(x_points)) + self.loboundx , torch.zeros(len(x_points)) + self.upboundx
        qLowerBound, qUpperBound =  torch.zeros(len(x_points)), torch.ones(len(x_points))
        # return pLowerBound, pUpperBound
        
        ptransformed_nodes = 0.5 * (pUpperBound[None,:] - pLowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (pUpperBound[None,:] + pLowerBound[None,:])
        ptransformed_weights = 0.5 *  (pUpperBound[None,:] - pLowerBound[None,:]) * guassweights[:, None]
        qtransformed_nodes = 0.5 * (qUpperBound[None,:] - qLowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (qUpperBound[None,:] + qLowerBound[None,:])
        qtransformed_weights = 0.5 *  (qUpperBound[None,:] - qLowerBound[None,:]) * guassweights[:, None]
         

        pNumerator = w[None, None, :]/(ptransformed_nodes[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qtransformed_nodes[:, :, None] - x_points[None, None, :])


        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]

        Integrandp =  ptransformed_weights[:,:, None, None]*(Bp[:,:,:,None])
        
        Bq = qNumerator/qDenominator[:,:,None]
        
        Integrandq =  qtransformed_weights[:,:, None, None]*(Bq[:,:,None,:])
        Result = torch.sum(Integrandp, dim = 0)*torch.sum(Integrandq, dim = 0)
        chiu = torch.ones(len(x_points))
        chiu[x_points < self.loboundu] = 0
        chiu[x_points > self.upboundu] = 0
        return (1/(self.upboundu - self.loboundu))*self.rate*chiu[:,None,None]*Result

    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
    
        chibounds = torch.ones(len(x_points))
        chibounds[x_points < self.loboundx] = 0
        chibounds[x_points > self.upboundx] = 0
        LowerBound, UpperBound = torch.zeros(len(x_points)), torch.ones(len(x_points))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        Numerator = w[None, None, :]/(transformed_nodes[:, :, None] - x_points[None, None, :])
    
        Denominator = torch.sum(Numerator, dim = 2 )
    
        Bp = transformed_weights[:,:, None]*Numerator/Denominator[:,:,None]
        # return Bp.shape
        Result = chibounds[:, None] * torch.sum(Bp, dim = 0)
        return self.rate*Result

##### Right with Encounter
class TransitionRightXY():
    def __init__(self, gamma, rt):
        self.TransitionType = 'Encounter'
        self.gamma = gamma
        self.rate = rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
        
        #### Limits:  0,x_points
        
        LowerBound, UpperBound = (x_points- self.gamma)/(1 - self.gamma), x_points
        LowerBound[LowerBound < 0] = 0
        
        ## inputs Bp = (x_points[l])/(1-gamma*x), input Bq: x
        
        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]
        
        
        ####### Structure: dim 0: gaussian | dim 1: j | dim 2: p | dim 3: q
        ## Integrand: def fintegran(x):
            # return barycentric_polynomial(x, q,w)*barycentric_polynomial((x_points[l])/(1-gamma*x), p, w)*(1/(1 - gamma*x))
        
            ## numerator = w[i] / (x - x_points[i])
        
        qInputs =1 - (x_points[None, :] - transformed_nodes)/(self.gamma*(1 - transformed_nodes))
        pInputs = transformed_nodes    
        # print('q', qInputs[qInputs < 0])
        # print('q', qInputs[qInputs >  1])
        # print(x_points)
        # print(qInputs[torch.isin(qInputs, x_points)] )
        pNumerator = w[None, None, :]/(pInputs[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qInputs[:, :, None] - x_points[None, None, :])

        
        # print('nom', pNumerator)
        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]
        
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        # print('den', pDenominator)
        Bq = qNumerator/qDenominator[:,:,None]
        
        AdditionalTerm = 1/(self.gamma*(1 - transformed_nodes))
        
        Integrand =  transformed_weights[:,:, None, None]*(Bp[:,:,:,None]*Bq[:,:,None,:]*AdditionalTerm[:, :, None,None])
        
        Result = torch.sum(Integrand, dim = 0)
        return self.rate*Result
    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))), torch.tensor(np.ones(len(x_points)))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        Numerator = w[None, None, :]/(transformed_nodes[:, :, None] - x_points[None, None, :])

        Denominator = torch.sum(Numerator, dim = 2 )

        Bp = transformed_weights[:,:, None]*Numerator/Denominator[:,:,None]

        Result = torch.sum(Bp, dim = 0)
        return self.rate*Result
##### Left with Encounter
class TransitionLeftXY():
    def __init__(self, gamma, rt):
        self.TransitionType = 'Encounter'
        self.gamma = gamma
        self.rate =  rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)
        
        
        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))), (1-x_points)/(self.gamma)
        UpperBound[UpperBound > 1] = 1

        
        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]
        

        
        pInputs = (x_points[None, :])/(1 - self.gamma*transformed_nodes)
        
        qInputs = transformed_nodes    
            
            
        pNumerator = w[None, None, :]/(pInputs[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qInputs[:, :, None] - x_points[None, None, :])
        
        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]
        
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bq = qNumerator/qDenominator[:,:,None]
        
        AdditionalTerm = 1/(1 - self.gamma*transformed_nodes)
        Integrand =  transformed_weights[:,:, None, None]*(Bp[:,:,:,None]*Bq[:,:,None,:]*AdditionalTerm[:, :, None,None])
    
        Result = torch.sum(Integrand, dim = 0)
        
        
        return self.rate*Result
    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))), torch.tensor(np.ones(len(x_points)))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        Numerator = w[None, None, :]/(transformed_nodes[:, :, None] - x_points[None, None, :])

        Denominator = torch.sum(Numerator, dim = 2 )

        Bp = transformed_weights[:,:, None]*Numerator/Denominator[:,:,None]

        Result = torch.sum(Bp, dim = 0)
        return self.rate*Result
##### Toward with Encounter
class TransitionTowardXY():
    def __init__(self, gamma, lim, rt):
        self.TransitionType = 'Encounter'
        self.gamma = gamma
        self.lim = lim
        self.rate = rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n, a,b)) #O(n)
        # SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

        #### Limits:  0, min(a, (a - xpoints)/(amma)

        LowerBound, UpperBound = (self.gamma*self.lim - x_points)/(self.gamma*self.lim),   torch.ones(n) 
        UpperBound[x_points > self.lim] = 0
        LowerBound[LowerBound < 0] = 0

        ## inputs Bp = x*F2 , input Bq: (t-u)/(gamma*(t-a))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        pInputs = (x_points[None,:] - self.gamma*self.lim*(1 - transformed_nodes))/(1 - self.gamma*(1 - transformed_nodes))
        qInputs = transformed_nodes  
            
        
        pNumerator = w[None, None, :]/(pInputs[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qInputs[:, :, None] - x_points[None, None, :])
        
        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        # print(pNumerator)

        # print(pNumerator[pNumerator != pNumerator])
        Bp = pNumerator/pDenominator[:,:,None]
        
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bp[pInputs[:, :, None] == x_points[None, None, :]] = 1 
        
        Bq = qNumerator/qDenominator[:,:,None]
        Bq[qInputs[:, :, None] == x_points[None, None, :]] = 1
        AdditionalTerm = 1/(1 - self.gamma*(1-transformed_nodes))
        
        Integrand =  transformed_weights[:,:, None, None]*(Bp[:,:,:,None]*Bq[:,:,None,:]*AdditionalTerm[:, :, None,None])
    
        LeftResult = torch.sum(Integrand, dim = 0)
        LeftResult[ x_points[:] > self.lim, :, :] = 0
        
        LowerBound, UpperBound =    torch.zeros(n),  (1 - x_points)/(self.gamma*(1 - self.lim))
        UpperBound[UpperBound > 1] = 1
        UpperBound[x_points < self.lim] = 0
        
        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]
        
        
        pInputs = (x_points[None, :] - self.gamma*self.lim*transformed_nodes)/(1 - self.gamma*(transformed_nodes))
        qInputs = transformed_nodes    
            
            
        pNumerator = w[None, None, :]/(pInputs[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qInputs[:, :, None] - x_points[None, None, :])
        
        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]
        Bp[pInputs[:, :, None] == x_points[None, None, :]] = 1
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bq = qNumerator/qDenominator[:,:,None]
        Bq[qInputs[:, :, None] == x_points[None, None, :]] = 1
        AdditionalTerm = 1/(1 - self.gamma*(transformed_nodes))
        Integrand =  transformed_weights[:,:, None, None]*(Bp[:,:,:,None]*Bq[:,:,None,:]*AdditionalTerm[:, :, None,None])
    
        RightResult = torch.sum(Integrand, dim = 0)
        
        return self.rate*(LeftResult + RightResult)
    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))), torch.tensor(np.ones(len(x_points)))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        Numerator = w[None, None, :]/(transformed_nodes[:, :, None] - x_points[None, None, :])

        Denominator = torch.sum(Numerator, dim = 2 )

        Bp = transformed_weights[:,:, None]*Numerator/Denominator[:,:,None]

        Result = torch.sum(Bp, dim = 0)
        return self.rate*Result
##### Away with Encounter
class TransitionAwayXY():
    def __init__(self, gamma, lim, rt):
        self.TransitionType = 'Encounter'
        self.gamma = gamma
        self.lim = lim
        self.rate = rt
    def MassAddition(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n, a,b)) #O(n)
        # SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

        #### Limits:  0, min(a, (a - xpoints)/(amma)
chi
        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))),  (self.lim - x_points)/(self.lim*self.gamma)
        UpperBound[UpperBound > 1] = 1
        UpperBound[UpperBound < 0] = 0

        ## inputs Bp = x*F2 , input Bq: (t-u)/(gamma*(t-a))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        pInputs = x_points[None,:]/(1 - self.gamma*transformed_nodes)
        qInputs = transformed_nodes  
            
        
        pNumerator = w[None, None, :]/(pInputs[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qInputs[:, :, None] - x_points[None, None, :])
        
        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]
        
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bp[pInputs[:, :, None] == x_points[None, None, :]] = 1 
        
        Bq = qNumerator/qDenominator[:,:,None]
         
        AdditionalTerm = 1/(1 - self.gamma*transformed_nodes)
        
        Integrand =  transformed_weights[:,:, None, None]*(Bp[:,:,:,None]*Bq[:,:,None,:]*AdditionalTerm[:, :, None,None])
    
        LeftResult = torch.sum(Integrand, dim = 0)

        

       
        
        LowerBound, UpperBound =    1 - (x_points - self.lim)/(self.gamma*(1 - self.lim)) , torch.ones(n)
        LowerBound[LowerBound > 1] = 1
        LowerBound[LowerBound < 0] = 0
        
        
        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]
        
        
        pInputs = (x_points[None, :] - self.gamma*(1-transformed_nodes))/(1 - self.gamma*(1-transformed_nodes))
        qInputs = transformed_nodes    
            
            
        pNumerator = w[None, None, :]/(pInputs[:, :, None] - x_points[None, None, :])
        
        qNumerator = w[None, None, :]/(qInputs[:, :, None] - x_points[None, None, :])
        
        pDenominator = torch.sum(pNumerator, dim = 2 )#[:, :, :, None]
        
        Bp = pNumerator/pDenominator[:,:,None]
        Bp[pInputs[:, :, None] == x_points[None, None, :]] = 1
        qDenominator = torch.sum(qNumerator, dim = 2 )#[:, :, :, None]
        
        Bq = qNumerator/qDenominator[:,:,None]
        
        AdditionalTerm = 1/(1 - self.gamma*(1 - transformed_nodes))
        Integrand =  transformed_weights[:,:, None, None]*(Bp[:,:,:,None]*Bq[:,:,None,:]*AdditionalTerm[:, :, None,None])
    
        RightResult = torch.sum(Integrand, dim = 0)
        
        return self.rate*(LeftResult + RightResult)
    def MassSubstraction(self, n, a = 0, b = 1):
        x_points = torch.tensor(ChebyshevNodes(n,a,b)) #O(n)
        SetXpoints = set(x_points)
        w = torch.tensor(barycentric_weights(x_points))
        qn = 100 # n #int(n/2)
        gaussnodes, guassweights = np.polynomial.legendre.leggauss(qn)
        gaussnodes, guassweights = torch.tensor(gaussnodes), torch.tensor(guassweights)

        LowerBound, UpperBound = torch.tensor(np.zeros(len(x_points))), torch.tensor(np.ones(len(x_points)))

        transformed_nodes = 0.5 * (UpperBound[None,:] - LowerBound[None,:]) * gaussnodes[:,None] + 0.5 *  (UpperBound[None,:] + LowerBound[None,:])
        transformed_weights = 0.5 *  (UpperBound[None,:] - LowerBound[None,:]) * guassweights[:, None]

        Numerator = w[None, None, :]/(transformed_nodes[:, :, None] - x_points[None, None, :])

        Denominator = torch.sum(Numerator, dim = 2 )

        Bp = transformed_weights[:,:, None]*Numerator/Denominator[:,:,None]

        Result = torch.sum(Bp, dim = 0)
        return self.rate*Result
###################################################################################################################################################################
## Kinetic System
#################################################################################################################################################################



class KineticSystem():
    def __init__(self, N, names = []):
        '''
        N: number of subsytems
        names: (OPtional) name of the subsytems
        '''

        self.N = N
        self.names = []
        self.TransitionRates = {i: [[ZeroFunc for _ in range(N)] for _ in range(N)] for i in range(N)}
        self.TransitionRatesNoEncounter = {i: [ZeroFunc for _ in range(N)] for i in range(N)}
        self.TransitionRatesChi = {i: [[ZeroFunc for _ in range(N)] for _ in range(N)] for i in range(N)}
        self.DensityAddition = {i: [] for i in range(N)} ### store subsystems Addition 
        self.DensitySubstraction = {i: [] for i in range(N)} ### store subsystems substraction 
        self.DensityAdditionNoEncounter = {i: [] for i in range(N)} ### store subsystems Addition 
        self.DensitySubstractionNoEncounter = {i: [] for i in range(N)} ### store subsystems substraction 
        self.DensityAdditionChi = {i: [] for i in range(N)} ### store subsystems Addition 
        self.DensitySubstractionChi = {i: [] for i in range(N)} ### store subsystems substraction 
        self.InterpolationDone = 0
    def AddInteraction(self, transitionrate, j,k,i = None):
        '''
        i: resulting subsytem
        j: first interagting particle
        k: second interacting particle
                            j --> i when j interacts with k.
        encouterrate: function | Encounter rate between j and k
        transitionrate: Transition Object | rule of transition
        '''
        if transitionrate.TransitionType == 'NoEncounter':
            self.TransitionRatesNoEncounter[k-1][j-1] = transitionrate
            self.DensityAdditionNoEncounter[k-1] += [(j-1)]
            self.DensitySubstraction[j-1] += [k-1]
        if transitionrate.TransitionType == 'Encounter':
                self.TransitionRates[i-1][j-1][k-1] = transitionrate
                self.DensityAddition[i-1] += [(j-1,k-1)]
                self.DensitySubstraction[j-1] += [k-1]
        if transitionrate.TransitionType == 'EncounterChi':
            self.TransitionRatesChi[i-1][j-1][k-1] = transitionrate
            self.DensityAdditionChi[i-1] += [(j-1,k-1)]
            self.DensitySubstractionChi[j-1] += [k-1]
    # def AddInteractionChi(self, j,k,i, transitionrate):
    #     '''
    #     i: resulting subsytem
    #     j: first interagting particle
    #     k: second interacting particle
    #                         j --> i when j interacts with k.
    #     encouterrate: function | Encounter rate between j and k
    #     transitionrate: Transition Object | rule of transition
    #     '''
    #     self.TransitionRatesChi[i-1][j-1][k-1] = transitionrate
    #     self.DensityAdditionChi[i-1] += [(j-1,k-1)]
    #     self.DensitySubstractionChi[j-1] += [k-1]
    def Solve(self, 
              initial_state,
              limit_time= 10, 
              num_time_points=10000,
              InterpolationNodesNumber = 20,
              InterpolationPoints = ChebyshevNodes,
              InterpolationBasis = barycentric_polynomial,
              ForceInterpolate = False):

        '''

        '''
        self.limit_time = limit_time
        self.num_time_points = num_time_points
        if self.InterpolationDone == 0 or ForceInterpolate == True:
            self.Interpolate(InterpolationNodesNumber = InterpolationNodesNumber, InterpolationPoints = InterpolationPoints, InterpolationBasis = InterpolationBasis)

        F0 = np.zeros(self.N*len(self.InterpolationNodes))
        for i in range(self.N):
            F0[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = initial_state[i](self.InterpolationNodes) 
        
        ts = np.linspace(0,limit_time,num_time_points+1)
        
        self.Fsolve = odeint(self.RHS, F0, ts)
        
        self.Solutions = {i: self.Fsolve[:,(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] for i in range(0,self.N)}
        
    def Interpolate(self,
                    InterpolationNodesNumber = 20,
                    InterpolationPoints = ChebyshevNodes,
                    InterpolationBasis = barycentric_polynomial):
        
        self.InterpolationBasis = InterpolationBasis
        
        self.InterpolationNodes = ChebyshevNodes(InterpolationNodesNumber)
        # print(len(self.InterpolationNodes))
        w =  barycentric_weights(self.InterpolationNodes)
        self.w = w
        self.MassAdditionWeights = [ [ [ 0 for k in range(self.N)] for h in  range(self.N)] for i in range(self.N)]
        self.MassAdditionWeightsChi = [ [ [ 0 for k in range(self.N)] for h in  range(self.N)] for i in range(self.N)]
        
        self.MassAdditionWeightsNoEncounter = [ [ 0 for h in  range(self.N)] for i in range(self.N)]
        self.MassSubstractionWeights =  [ [ 0
                                   for j in range(self.N)] 
                                       for i in range(self.N) ]
        self.MassSubstractionWeightsNoEncounter = [0 for i in range(self.N) ]
        self.MassSubstractionWeightsChi =  [ [ 0
                                   for j in range(self.N)] 
                                       for i in range(self.N) ]
        ### Creating Interpolation:
        for i in range(self.N):
            for Tuple in  self.DensityAddition[i]:
                h,k = Tuple
                
                self.MassAdditionWeights[i][h][k] =  self.TransitionRates[i][h][k].MassAddition(InterpolationNodesNumber, 0, 1)
                self.MassSubstractionWeights[h][k] = self.TransitionRates[i][h][k].MassSubstraction(InterpolationNodesNumber)
            for Tuple in  self.DensityAdditionChi[i]:
                h,k = Tuple
                
                self.MassAdditionWeightsChi[i][h][k] =  self.TransitionRatesChi[i][h][k].MassAddition(InterpolationNodesNumber, 0, 1)
                
                
                self.MassSubstractionWeightsChi[h][k] = self.TransitionRatesChi[i][h][k].MassSubstraction(InterpolationNodesNumber, 0, 1)
            
            for h in  self.DensityAdditionNoEncounter[i]:
                
                self.MassAdditionWeightsNoEncounter[i][h] =  self.TransitionRatesNoEncounter[i][h].MassAddition(InterpolationNodesNumber, 0, 1)
                
                
                self.MassSubstractionWeightsNoEncounter[h] = self.TransitionRatesNoEncounter[i][h].MassSubstraction(InterpolationNodesNumber, 0, 1)
                
            
        
    
    
        self.BassisIntegral = BasisIntegralGenerator(InterpolationNodesNumber)
        self.InterpolationDone = 1 
    # def rk45(self, rhs, y0, t_span, **kwargs):
    def rk45(self, rhs, y0, t_span, mass_tol=1e-10):
        """
        A custom Runge-Kutta 4(5) integrator with mass preservation and non-negativity.
    
        Parameters:
        - rhs: callable
            The right-hand side function of the ODE, dy/dt = rhs(t, y).
        - y0: array_like
            Initial values for the system.
        - t_span: array_like
            Time array where solutions are required.
        - mass_tol: float
            Tolerance for mass preservation.
    
        Returns:
        - solutions: np.ndarray
            Array of solutions with shape (len(t_span), len(y0)).
        """
        # Butcher tableau for Dormand-Prince RK45
        A = [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]
        B = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
        C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1]
    
        y = np.array(y0, dtype=float)
        total_mass = np.sum(y)
        solutions = [y.copy()]
        t = t_span[0]
    
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]
            k = [np.zeros_like(y) for _ in range(6)]
    
            # Calculate RK stages
            for j in range(6):
                t_j = t + C[j] * dt
                y_j = y + dt * sum(A[j][m] * k[m] for m in range(len(A[j])))
                k[j] = rhs( y_j, t_j)
    
            # Update solution
            y = y + dt * sum(B[j] * k[j] for j in range(6))
    
            # Ensure non-negativity
            y[y < 0] = 0
    
            # Adjust for mass preservation
            current_mass = np.sum(y)
            if abs(current_mass - total_mass) > mass_tol and current_mass > 0:
                y *= total_mass / current_mass
    
            solutions.append(y.copy())
            t += dt
    
        return np.array(solutions)
    def RHS(self, F0,t):
            
            F0s = [torch.tensor(F0[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))]) for i in range(0,self.N)]
            
            dFAdd = [np.zeros(len(self.InterpolationNodes)) for i in range(self.N)]
            dFSub = [np.zeros(len(self.InterpolationNodes)) for i in range(self.N)]
            
            for i in range(self.N):
                for Tuple in  self.DensityAddition[i]:
                    h,k = Tuple
                    dFAdd[i] += torch.sum(self.MassAdditionWeights[i][h][k]*F0s[h][None, :, None]*F0s[k][None, None, :], dim = (1,2)).numpy()
                for Tuple in  self.DensityAdditionChi[i]:
                    h,k = Tuple
                    dFAdd[i] += torch.sum(self.MassAdditionWeightsChi[i][h][k]*F0s[h][None, :, None]*F0s[k][None, None, :], dim = (1,2)).numpy()
                    dFSub[h] += torch.sum(F0s[h][:, None]* self.MassSubstractionWeightsChi[h][k]*F0s[k][None, :], dim = 1).numpy()
                for q in set(self.DensitySubstraction[i]):
                    dFSub[i] += torch.sum(F0s[i][:, None]* self.MassSubstractionWeights[i][q]*F0s[q][None, :], dim = 1).numpy()

                for h in self.DensityAdditionNoEncounter[i]:
                    dFAdd[i] += torch.sum(F0s[h][None,:]* self.MassAdditionWeightsNoEncounter[i][h], dim = 1).numpy()
                    dFSub[h] += F0s[h].numpy()*self.MassSubstractionWeightsNoEncounter[h].numpy()
            dFAddArray = np.zeros(len(F0))
            dFSubArray = np.zeros(len(F0))
            basisintegral = np.zeros(len(F0))
            for i in range(0,self.N):
                dFAddArray[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = dFAdd[i]
                dFSubArray[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = dFSub[i]
                basisintegral[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = self.BassisIntegral
            
            C = np.sum(dFAddArray*basisintegral)/np.sum(dFSubArray*basisintegral)
            dF = dFAddArray - C*dFSubArray
        
            return dF
    def Ccalculator(self, F0fct):
        nodes = self.N*list(self.InterpolationNodes)
        # print(self.InterpolationNodes)

        F0 = F0fct(np.array(nodes))
        F0s = [torch.tensor(F0[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))]) for i in range(0,self.N)]
        sol = self.RHS(F0,0)
        basis = np.zeros(len(F0))
        for i in range(0,self.N):
            basis[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = self.BassisIntegral
        dFAdd = [np.zeros(len(self.InterpolationNodes)) for i in range(self.N)]
        dFSub = [np.zeros(len(self.InterpolationNodes)) for i in range(self.N)]    
        for i in range(self.N):
            for Tuple in  self.DensityAddition[i]:
                h,k = Tuple
                dFAdd[i] += torch.sum(self.MassAdditionWeights[i][h][k]*F0s[h][None, :, None]*F0s[k][None, None, :], dim = (1,2)).numpy()
            
            for q in set(self.DensitySubstraction[i]):
                dFSub[i] += torch.sum(F0s[i][:, None]* self.MassSubstractionWeights[i][q]*F0s[q][None, :], dim = 1).numpy()

        

        dFAddArray = np.zeros(len(F0))
        dFSubArray = np.zeros(len(F0))
        basisintegral = np.zeros(len(F0))
        for i in range(0,self.N):
            dFAddArray[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = dFAdd[i]
            dFSubArray[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = dFSub[i]
            basisintegral[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = self.BassisIntegral
        
        C = np.sum(dFAddArray*basisintegral)/np.sum(dFSubArray*basisintegral)
        return C
    def CheckNumericalMassPreservation(self, trial_num):

        for i in range(0,trial_num):
            F0 = np.random.randint(0,10,self.N*len(self.InterpolationNodes))
            sol = self.RHS(F0,0)
            basis = np.zeros(len(F0))
            for i in range(0,self.N):
                basis[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = self.BassisIntegral
            print(np.sum(sol*basis))

    def CheckNumericalMassPreservationSol(self, tlist):
        basis = np.zeros(len(self.Fsolve[0,:]))
        mass = []
        for i in range(0,self.N):
            basis[(i)*(len(self.InterpolationNodes)): (i+1)*(len(self.InterpolationNodes))] = self.BassisIntegral
        for t in tlist:
                mass += [np.sum(self.Fsolve[int(np.floor(t*(self.num_time_points-1)/self.limit_time)),:]*basis)]
        return mass
    def SpatialInterpolator(self, x, F):
        w =  barycentric_weights(self.InterpolationNodes)
        Interp = 0
        for i in range(0,len(self.InterpolationNodes)):
            Interp += F[i]*self.InterpolationBasis(x,self.InterpolationNodes,i,w)
        return Interp
    def InterpolateSolution(self, i, x,t):
        
        return self.SpatialInterpolator(x, self.Solutions[i][int(np.floor(t*(self.num_time_points-1)/self.limit_time)),:len(self.InterpolationNodes)])
    def Average(self,i,t):
        mass = integrate.quad(lambda x: self.InterpolateSolution(i-1, x, t),0,1 )[0]
        if mass < 10**(-16):
            return 0
        else:
            return integrate.quad(lambda x: x*self.InterpolateSolution(i-1, x, t),0,1 )[0]/mass
    def Mass(self,i,t):
        return integrate.quad(lambda x: self.InterpolateSolution(i-1, x, t),0,1 )[0]
    def MultiPlot(self,i, ts, name = '', TimeLabel = False):
        plt.style.use('tableau-colorblind10')
        
        Pts = np.linspace(0,1,100+1)
        pallette = sns.color_palette("colorblind", n_colors=8)
        maxsol = np.max(self.Solutions[i-1])
        for t in ts:
            plt.figure(figsize=(8, 8))
            plt.plot(Pts,  [self.InterpolateSolution(i-1, p, t) for p in Pts], 
                     linewidth=4,
                     linestyle = 'solid', 
                     color = pallette[i],
                     label = 't='+str(round(t,2)))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            
            # plt.gca().relim()  # Recalculate limits
            plt.gca().autoscale_view() 
            plt.grid(linestyle='dotted')
            # print(np.max(self.Solutions[i-1]))
            plt.ylim(0,max(np.max(self.Solutions[i-1]),1))
            if TimeLabel:
                plt.legend(ncol=1, loc=2) # 9 means top center | 2 means top left

            plt.tight_layout()

            if name != '':
                plt.savefig(name+'f'+str(i)+'t'+str(t)+'.jpg', dpi = 500)
            plt.show()
    def MultiPlotForAnimation(self,i, ts, name = '', TimeLabel = False):
        plt.style.use('tableau-colorblind10')
        
        Pts = np.linspace(0,1,100+1)

        pallette = sns.color_palette("colorblind", n_colors=8)

        for t in ts:

            plt.figure(figsize=(8, 8))
            plt.plot(Pts,  [self.InterpolateSolution(i-1, p, t) for p in Pts], 
                     linewidth=4,
                     linestyle = 'solid', 
                     color = pallette[i],
                     label = 't='+str(round(t,2)))
  
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
      
            plt.ylim(0,max(np.max(self.Solutions[i-1]),1))
            plt.gca().relim()  # Recalculate limits
            plt.gca().autoscale_view() 
            plt.grid(linestyle='dotted')
            if TimeLabel:
                plt.legend(ncol=1, loc=2) # 9 means top center | 2 means top left
            
            plt.tight_layout()
            
            if name != '':
                plt.savefig(name+'f'+str(i)+'t'+str(t)+'.jpg', dpi = 500)
            plt.clf()
    def Animate(self, i, name, duration = 5, fps = 24, TimeLabel = True):
        if not os.path.exists('tempframes'):
            os.makedirs('tempframes') 
        print('Generating frames ....', end = '\r')
        total_frames_needed = duration*fps
        ts = np.linspace(0, self.limit_time, total_frames_needed)
        self.MultiPlotForAnimation(i, ts,name = 'tempframes/', TimeLabel = TimeLabel)
        
        images = [os.path.join('tempframes', f"f{i}t{t}.jpg") for t in ts]
        print('Generating Gif ....', end = '\r')
        # Read and store frames
        frames = []
        for img_path in images:
            if os.path.exists(img_path):
                frames.append(imageio.imread(img_path))
            else:
                print(f"Warning: {img_path} not found!")
        
        # Save as GIF
        imageio.mimsave(name + ".gif", frames, fps=fps)

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for notebooks / interactive mode
            current_dir = os.getcwd()
        folder_path = os.path.join(current_dir, 'tempframes')
        shutil.rmtree(folder_path)
        print('Complete')
    def PlotMass(self,i, name = ''):
        plt.style.use('tableau-colorblind10')
        Pts = np.linspace(0,1,100+1)
        
        Massi = torch.sum(self.BassisIntegral[None,:]*self.Solutions[i-1], axis = 1)
        plt.plot(np.linspace(0,self.limit_time,self.num_time_points+1),  Massi, 
                     linewidth=4,
                     linestyle = 'solid', 
                     )
        plt.gca().relim()  # Recalculate limits
        plt.gca().autoscale_view() 
        plt.grid(linestyle='dotted')
        plt.xlabel('t')
        # plt.legend(ncol=1, loc=2) # 9 means top center | 2 means top left
        plt.title('f'+str(i))
        plt.tight_layout()
        plt.legend(prop={'size': 15})
        if name != '':
             plt.savefig(name+'f'+str(i)+'.jpg', dpi = 500)
        plt.show()





