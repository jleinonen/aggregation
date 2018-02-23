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

import numpy
from numpy import array
from scipy import ndimage

def generate_dendrite(alpha, beta, gamma, grid_size=1000, num_iter=10000):
    """Generate a 2D dendrite.

    This is a numpy vectorized implementation of the algorithm by Reiter
    (2005), https://doi.org/10.1016/j.chaos.2004.06.071

    Args:
        alpha, beta, gamma: Parameters of the algorithm, refer to the Reiter
            (2005) paper for details.
        grid_size: The number of elements per dimension in the 2D grid.
        num_iter: Number of iterations.
    """

    grid = numpy.zeros((grid_size,grid_size))
    x = numpy.tile(numpy.linspace(-grid_size/2.0,grid_size/2.0,grid_size),
        (grid_size,1))
    y = numpy.tile(numpy.linspace(-grid_size/2.0,grid_size/2.0,grid_size),
        (grid_size,1)).T
    boundary_mask = (x**2+y**2 >= (grid_size/2.0)**2)
    grid[:] = beta
    grid[grid_size/2,grid_size/2] = 1.0
   
    it = 0

    even = ((numpy.arange(grid_size) % 2) == 0)
    odd = ~even

    neighbor_k_even = array([[1.0,1.0,0.0],
                            [1.0,1.0,1.0],
                            [1.0,1.0,0.0]])
    neighbor_k_odd = array([[0.0,1.0,1.0],
                           [1.0,1.0,1.0],
                           [0.0,1.0,1.0]])
                                           
    nonrecp_avg_k_even = array([[alpha/12.0, alpha/12.0,    0.0       ],
                               [alpha/12.0, 1.0-alpha/2.0, alpha/12.0],
                               [alpha/12.0, alpha/12.0,    0.0       ]])  
    nonrecp_avg_k_odd = array([[0.0,        alpha/12.0,    alpha/12.0],
                              [alpha/12.0, 1.0-alpha/2.0, alpha/12.0],
                              [0.0,        alpha/12.0,    alpha/12.0]])
                            
   
    while it < num_iter:
        ice = (grid >= 1.0)
        receptive = (ndimage.convolve(ice,neighbor_k_even,mode='constant') > 0.0)
        receptive[odd,:] = (ndimage.convolve(ice,neighbor_k_odd,mode='constant')[odd,:] > 0.0)
         
        nonrecp = grid.copy()
        nonrecp[receptive] = 0.0
        nonrecp_even = ndimage.convolve(nonrecp,nonrecp_avg_k_even,mode='constant')
        nonrecp_odd = ndimage.convolve(nonrecp,nonrecp_avg_k_odd,mode='constant')
        nonrecp[even,:] = nonrecp_even[even,:]
        nonrecp[odd,:] = nonrecp_odd[odd,:]      
        
        grid[~receptive] = 0.0
        grid[grid != 0.0] += gamma
        grid += nonrecp
        
        grid[boundary_mask] = beta
        
        it += 1
      
    return grid
