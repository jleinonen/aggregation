import numpy
from numpy import array
from scipy import ndimage

def generate_dendrite(alpha, beta, gamma, grid_size=1000, num_iter=10000):
   grid = numpy.zeros((grid_size,grid_size))
   x = numpy.tile(numpy.linspace(-grid_size/2.0,grid_size/2.0,grid_size),(grid_size,1))
   y = numpy.tile(numpy.linspace(-grid_size/2.0,grid_size/2.0,grid_size),(grid_size,1)).T
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
