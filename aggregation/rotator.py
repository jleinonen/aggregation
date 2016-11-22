import numpy as np
from numpy import array, random, pi
from scipy.integrate import cumtrapz
from scipy.interpolate import splrep, splev

class Rotator(object):

    @staticmethod
    def rotation_matrix(alpha, beta, psi):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cp = np.cos(psi)
        sp = np.sin(psi)

        Rb = array([[1.0, 0.0, 0.0],
                    [0.0, cb,  -sb],
                    [0.0, sb,  cb ]])
        Ra = array([[ca , -sa, 0.0],
                    [sa , ca , 0.0],
                    [0.0, 0.0, 1.0]])
        Rp = array([[cp , -sp, 0.0],
                    [sp , cp , 0.0],
                    [0.0, 0.0, 1.0]])

        return np.dot(np.dot(Ra,Rb),Rp)


    def __init__(self):
        pass

    def rotate(X):
        return X


class UniformRotator(Rotator):
    def rotate(self,X):
        alpha = random.rand() * 2*pi

        beta = np.arccos(1.0-2*random.rand())
        psi = random.rand() * 2*pi

        R = Rotator.rotation_matrix(alpha,beta,psi)
        return np.dot(R,X)


class HorizontalRotator(Rotator):
    def rotate(self,X):
        alpha = random.rand() * 2*pi
        R = Rotator.rotation_matrix(alpha,0.0,0.0)
        return np.dot(R,X)
    
    
class SamplePDF(object):
    def __init__(self, pdf, a, b, num_points=1024):
        x = np.linspace(a, b, num_points)        
        y = pdf(x)
        Y = np.hstack((0, cumtrapz(y,x)))
        Y /= Y[-1]
        self.interp = splrep(Y, x, s=0)
        
    def __call__(self):
        return self.rvs()
        
    def rvs(self):
        return splev(random.rand(), self.interp, der=0)
        
        

class PartialAligningRotator(Rotator):
    def __init__(self, exp_sig_deg=40, random_flip=False):
        self.exp_sig = exp_sig_deg * pi/180
        self.beta_sample = SamplePDF(
            lambda x: np.sin(x)*np.exp(-x**2/(2*self.exp_sig)), 0, pi)    
        self.random_flip = random_flip
    
    def rotate(self,X):
        alpha = random.rand() * 2*pi
        beta = self.beta_sample()
        R = Rotator.rotation_matrix(alpha,beta,0.0)
        X = np.dot(R, X)
        if self.random_flip and (random.rand() > 0.5):            
            X[2,:] = -X[2,:]
            X[1,:] = -X[1,:]
        return X
