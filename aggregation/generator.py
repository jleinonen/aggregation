import numpy as np
import crystal
import rotator
from misc import doc_inherit

class Generator(object):
   """Crystal generator.

   Crystal generators create 3-D volumetric models of ice crystals based
   on the crystal geometries found in the cystal module.

   This is a base class for all generators, which should implement the
   "generate" method.
   """
   def __init__(self):
      pass
   
   def generate(self):
      """Create a volume-element realization of the crystal.
      """
      pass
   

class MonodisperseGenerator(Generator):
   """Crystal generator for monodisperse particles.

   Crystal generators create 3-D volumetric models of ice crystals based
   on the crystal geometries found in the cystal module.

   Constructor args:
      crystal: A subclass of Crystal found in the "crystal" module.
      rot: A rotator found in the "rotator" module.
      grid_res: The grid spacing of the volume elements on a Cartesian grid.
   """ 
   def __init__(self, crystal, rot, grid_res):
      self.crystal = crystal
      self.rot = rot
      max_r = crystal.max_radius()
      #round up to nearest multiple of grid_res
      max_r += grid_res - max_r%grid_res 
      
      grid_bounds = [-max_r, max_r, -max_r, max_r, -max_r, max_r]
      self.grid_res = grid_res
      (self.x,self.y,self.z) = np.mgrid[
         grid_bounds[0]:grid_bounds[1]+grid_res*0.001:grid_res,
         grid_bounds[2]:grid_bounds[3]+grid_res*0.001:grid_res,
         grid_bounds[4]:grid_bounds[5]+grid_res*0.001:grid_res
      ]

   def generate(self):
      """Create a volume-element realization of the crystal.

      Returns:
         A volume-element realization of the crystal as a (N,3) array.
      """
      inside = self.crystal.is_inside(self.x,self.y,self.z)
      X = np.vstack((self.x[inside],self.y[inside],self.z[inside]))
      return self.rot.rotate(X)
