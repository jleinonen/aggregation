import crystal
import rotator
import numpy
from numpy import array   

class Generator:
   def __init__(self):
      pass
   
   def generate(self):
      pass
   

class MonodisperseGenerator(Generator):
   
   def __init__(self, crystal, rot, grid_res):      
      Generator.__init__(self)
      self.crystal = crystal
      self.rot = rot
      max_r = crystal.max_radius()
      #round up to nearest multiple of grid_res
      max_r += grid_res - max_r%grid_res 
      
      grid_bounds = [-max_r, max_r, -max_r, max_r, -max_r, max_r]
      self.grid_res = grid_res
      (self.x,self.y,self.z) = numpy.mgrid[grid_bounds[0]:grid_bounds[1]+grid_res*0.001:grid_res,\
                                           grid_bounds[2]:grid_bounds[3]+grid_res*0.001:grid_res,\
                                           grid_bounds[4]:grid_bounds[5]+grid_res*0.001:grid_res]
                                                 
   def generate(self):
      inside = self.crystal.is_inside(self.x,self.y,self.z)
      X = numpy.vstack((self.x[inside],self.y[inside],self.z[inside]))      
      return self.rot.rotate(X)
      
