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

import aggregate, crystal, rotator, generator
from numpy import array, random
import numpy
import cPickle as pickle
from scipy import stats


def monodisp_dendrite(N=5,grid=None,align=True):
   #cry = crystal.Column(0.3e-3)
   cry = crystal.Dendrite(0.5e-3,alpha=0.705,beta=0.5,gamma=0.0001,num_iter=2500,hex_grid=grid)
   rot = rotator.UniformRotator()
   gen = generator.MonodisperseGenerator(cry, rot, 0.02e-3)   
   
   agg = [aggregate.Aggregate(gen, levels=5) for i in xrange(N)]
   
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
            print i, j
            agg_top = agg[i] if (m_r[i] > m_r[j]) else agg[j]
            agg_btm = agg[i] if (m_r[i] <= m_r[j]) else agg[j]
            collision = agg_top.add_particle(particle=agg_btm.X,required=False)
            if collision:
               if align:
                  agg_top.align()
               else:
                  agg_top.rotate(rot)
               agg.pop(i if (m_r[i] <= m_r[j]) else j)            
      
   print aggregate.t_i, aggregate.t_o   
   
   if align:
      agg[0].align()
   agg[0].rotate(rotator.HorizontalRotator())
   
   return agg[0]


def gen_monodisp(N_range=(1,101)):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(*N_range):
      agg = monodisp_dendrite(N=N,grid=grid)      
      numpy.savetxt("monodisp/monod_"+str(N)+".dat",agg.grid(),fmt="%d")
      
def gen_monodisp_nonaligned(N_range=(1,101)):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(*N_range):
      agg = monodisp_dendrite(N=N,grid=grid,align=False)      
      numpy.savetxt("monodisp/monod_"+str(N)+".dat",agg.grid(),fmt="%d")
      
def gen_monodisp_Nmon(N=10,N0=0,Nmon=50):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(N0,N):
      agg = monodisp_dendrite(N=Nmon,grid=grid,align=False)      
      numpy.savetxt("monodisp_" + str(Nmon) + "/monod_"+str(N)+".dat",agg.grid(),fmt="%d")
      
      
def gen_monodisp_single():
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(300):
      agg = monodisp_dendrite(N=1,grid=grid,align=False)      
      numpy.savetxt("monodisp_single/monod_"+str(N)+".dat",agg.grid(),fmt="%d")  
      

def polydisp_dendrite(N=5,grid=None,align=True):
   #cry = crystal.Column(0.3e-3)
   
   agg = []
   psd = stats.expon(scale=1.0e-3)
   rot = rotator.UniformRotator()
   
   for i in xrange(N):
      D = 1e3
      while D > 0.3e-2 or D < 0.2e-3:
         D = psd.rvs()
      print "D: " + str(D)
      cry = crystal.Dendrite(D,alpha=0.705,beta=0.5,gamma=0.0001,num_iter=2500,hex_grid=grid)   
      gen = generator.MonodisperseGenerator(cry, rot, 0.02e-3) 
      agg.append(aggregate.Aggregate(gen, levels=5))
   
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
            print i, j
            agg_top = agg[i] if (m_r[i] > m_r[j]) else agg[j]
            agg_btm = agg[i] if (m_r[i] <= m_r[j]) else agg[j]
            collision = agg_top.add_particle(particle=agg_btm.X,required=True)
            if collision:
               if align:
                  agg_top.align()
               else:
                  agg_top.rotate(rot)
               agg.pop(i if (m_r[i] <= m_r[j]) else j)            
      
   print aggregate.t_i, aggregate.t_o   
   
   if align:
      agg[0].align()
   agg[0].rotate(rotator.HorizontalRotator())
   
   return agg[0]


def gen_polydisp(N_range=(1,101)):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(*N_range):
      agg = polydisp_dendrite(N=N,grid=grid)      
      numpy.savetxt("polydisp/polyd_"+str(N)+".dat",agg.grid(),fmt="%d")
      
def gen_polydisp_nonaligned(N_range=(1,101)):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(*N_range):
      agg = polydisp_dendrite(N=N,grid=grid,align=False)      
      numpy.savetxt("polydisp_nonalign/polyd_"+str(N)+".dat",agg.grid(),fmt="%d")
      
def gen_polydisp_single():
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(300):
      agg = polydisp_dendrite(N=1,grid=grid,align=False)      
      numpy.savetxt("polydisp_single/polyd_"+str(N)+".dat",agg.grid(),fmt="%d")        

def gen_polydisp_Nmon(N=10,N0=0,Nmon=50):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(N0,N):
      agg = polydisp_dendrite(N=Nmon,grid=grid,align=False)      
      numpy.savetxt("polydisp_" + str(Nmon) + "/polyd_"+str(N)+".dat",agg.grid(),fmt="%d")
      
      
def monodisp_pseudo(N=5,grid=None,sig=1.0):
   cry = crystal.Dendrite(0.5e-3,alpha=0.705,beta=0.5,gamma=0.0001,num_iter=2500,hex_grid=grid)
   rot = rotator.UniformRotator()
   gen = generator.MonodisperseGenerator(cry, rot, 0.02e-3)   
   
   """
   p_agg = aggregate.PseudoAggregate(gen, sig=0.1e-2)
   rho_i = 916.7 #kg/m^3
   N_dip = p_agg.grid().shape[0]
   m = 0.02e-3**3 * N_dip * N * rho_i
   sig = (m/20.3)**(1.0/2.35)
   print N_dip, sig
   """
   
   p_agg = aggregate.PseudoAggregate(gen, sig=sig)
   aggs = [aggregate.Aggregate(gen, levels=5) for i in xrange(N-1)]
   for agg in aggs:
      p_agg.add_particle(particle=agg.X,required=False)
   return p_agg
      
      
def gen_monodisp_pseudo(N_range=(1,101)):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(*N_range):
      agg = monodisp_pseudo(N=N,grid=grid)      
      numpy.savetxt("monodisp_pseudo/monod_"+str(N)+".dat",agg.grid(),fmt="%d")
      
def gen_monodisp_pseudo_Nmon(N=10,N0=0,Nmon=50):
   grid = pickle.load(file("dendrite_grid.dat"))
   for N in xrange(N0,N):
      agg = monodisp_pseudo(N=Nmon,grid=grid,sig=0.012)      
      numpy.savetxt("monodisp_pseudo_" + str(Nmon) + "/monod_"+str(N)+".dat",agg.grid(),fmt="%d")
