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
from numpy import random, array
import aggregate, generator, rotator, crystal
import cPickle as pickle

def monodisp_demo(N=5):
   #cry = crystal.Plate(0.3e-3)
   #cry = crystal.Rosette(0.6e-3)
   cry = crystal.Spheroid(0.6e-3,0.6)
   rot = rotator.UniformRotator()
   gen = generator.MonodisperseGenerator(cry, rot, 0.01e-3)   
   
   agg = aggregate.Aggregate(gen)
   
   for i in xrange(N-1):   
      print(i)
      agg.add_particle(required=True, pen_depth=0.02e-3)
      agg.align()
      
   return agg
   

def monodisp_demo2(N=5):
   #cry = crystal.Column(0.3e-3)
   cry = crystal.Dendrite(0.3e-3,0.705,0.5,0.0001,num_iter=2500)
   rot = rotator.UniformRotator()
   gen = generator.MonodisperseGenerator(cry, rot, 0.01e-3)   
   
   agg = [aggregate.Aggregate(gen) for i in xrange(N)]
   
   aggregate.t_i = 0
   aggregate.t_o = 0

   while len(agg) > 1:
      r = array([((a.extent[0][1]-a.extent[0][0])+(a.extent[1][1]-a.extent[1][0]))/4.0 for a in agg])
      m_r = numpy.sqrt(array([a.X.shape[0] for a in agg])/r)
      r_mat = (numpy.tile(r,(len(agg),1)).T+r)**2
      mr_mat = abs(numpy.tile(m_r,(len(agg),1)).T - m_r)
      p_mat = r_mat * mr_mat      
      p_max = p_mat.max()
      p_mat /= p_mat.max()
      collision = False
      while not collision:
         i = random.randint(len(agg))
         j = random.randint(len(agg))
         rnd = random.rand()
         if rnd < p_mat[i][j]:             
            print(i, j)
            agg_top = agg[i] if (m_r[i] > m_r[j]) else agg[j]
            agg_btm = agg[i] if (m_r[i] <= m_r[j]) else agg[j]
            collision = agg_top.add_particle(particle=agg_btm.X,required=True)
            agg_top.align()
            agg.pop(i if (m_r[i] <= m_r[j]) else j)            
      
   print(aggregate.t_i, aggregate.t_o)
   return agg[0]
   
   

