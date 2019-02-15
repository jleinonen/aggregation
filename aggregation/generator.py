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
from . import crystal
from . import rotator


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
   on the crystal geometries found in the crystal module.

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
