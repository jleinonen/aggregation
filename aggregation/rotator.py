"""
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy import array, random, pi
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


class Rotator(object):
    """Rotator object.

    A Rotator instance is used to rotate a set of 3D coordinates around
    the [0,0,0] point. All Rotator subclasses must implement the "rotate"
    member function.
    """

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
        """Rotate the coordinates around the [0,0,0] point.

        Args:
            The (N,3) array of coordinates.
        """
        return X


class UniformRotator(Rotator):
    """Uniformly random rotation.
    """

    def rotate(self,X):
        """Rotate the coordinates around the [0,0,0] point.

        Args:
            The (N,3) array of coordinates.
        """
        alpha = random.rand() * 2*pi

        beta = np.arccos(1.0-2*random.rand())
        psi = random.rand() * 2*pi

        R = Rotator.rotation_matrix(alpha,beta,psi)
        return np.dot(R,X)


class HorizontalRotator(Rotator):
    """Uniformly random rotation around the z axis only.
    """

    def rotate(self,X):
        """Rotate the coordinates around the [0,0,0] point.

        Args:
            The (N,3) array of coordinates.
        """
        alpha = random.rand() * 2*pi
        R = Rotator.rotation_matrix(alpha,0.0,0.0)
        return np.dot(R,X)
    
    
class SamplePDF(object):
    """Generate samples from a PDF given by a function.

    Constructor args:
        pdf: Function giving the PDF.
        a,b: Start and end of the interval at which the function is defined.
        num_points: Number of points used in the numerical evaluation of the
            function.
    """

    def __init__(self, pdf, a, b, num_points=1024):
        x = np.linspace(a, b, num_points)
        y = pdf(x)
        Y = np.hstack((0, cumtrapz(y,x)))
        Y /= Y[-1]
        self.interp = interp1d(Y, x, kind='linear')
        
    def __call__(self):
        return self.rvs()
        
    def rvs(self):
        """Samples from the PDF.

        Returns:
            A random sample from the specified PDF.
        """
        return self.interp(random.rand())
        
        

class PartialAligningRotator(Rotator):
    """Rotation into a Gaussian-weighted random orientation.

    This rotates the coordinates using a uniform PDF around the z axis and
    using a Gaussian-weighted canting angle for the tilt from the z axis.
    This can be used together with the Aggregate.align function to produce
    partially horizontally aligned aggregates.

    Constructor args:
        exp_sig_deg: The standard deviation of the canting angle, in degrees.
    """

    def __init__(self, exp_sig_deg=40, random_flip=False):
        self.exp_sig = exp_sig_deg * pi/180
        self.beta_sample = SamplePDF(
            lambda x: np.sin(x)*np.exp(-0.5*(x/self.exp_sig)**2), 0, pi)
        self.random_flip = random_flip
    
    def rotate(self,X):
        """Rotate the coordinates around the [0,0,0] point.

        Args:
            The (N,3) array of coordinates.
        """
        alpha = random.rand() * 2*pi
        beta = self.beta_sample()
        R = Rotator.rotation_matrix(alpha,beta,0.0)
        X = np.dot(R, X)
        if self.random_flip and (random.rand() > 0.5):
            X[2,:] = -X[2,:]
            X[1,:] = -X[1,:]
        return X
