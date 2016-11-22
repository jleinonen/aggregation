from bisect import bisect_left
import numpy as np
from numpy import random
from scipy import linalg, stats
import generator
from index2D import Index2D
from matplotlib import pyplot, colors
import math


def min_z_separation(elems,ref_elem,grid_res_sqr):
    elems = np.array(list(elems))
    if not len(elems):  
        return np.inf
    x_sep_sqr = ((elems[:,:2]-ref_elem[:2])**2).sum(1)
    match_possible = (x_sep_sqr < grid_res_sqr)
    z_sep = np.empty(match_possible.shape)
    z_sep[match_possible] = elems[match_possible,2] - ref_elem[2] - \
        np.sqrt(grid_res_sqr-x_sep_sqr[match_possible])
    z_sep[~match_possible] = np.inf            
    return z_sep.min()

current = None
other = None

class Aggregate(object):
    def __init__(self, generator, ident=0):
        self._generator = generator
        self.grid_res = generator.grid_res            
        self.X = self._generator.generate().T
        self.ident = np.empty(self.X.shape[0], dtype=np.uint32)
        self.ident.fill(ident)
        self.update_extent()


    def update_extent(self):
        x = self.X[:,0]
        y = self.X[:,1]
        z = self.X[:,2]
        self.extent = [[x.min(), x.max()], [y.min(), y.max()], 
            [z.min(), z.max()]]


    def vertical_projected_area(self):
        ext = self.extent
        x = (self.X[:,0]-ext[0][0]) / self.grid_res
        y = (self.X[:,1]-ext[1][0]) / self.grid_res
        x_max = int(round(x.max()))
        y_max = int(round(y.max()))

        proj_grid = np.zeros((x_max+1,y_max+1), dtype=np.uint8)
        proj_grid[x.round().astype(int), y.round().astype(int)] = 1

        return proj_grid.sum() * self.grid_res**2

                  
    def add_particle(self, particle=None, ident=None, required=False, pen_depth=0.0):
      
        if particle is None:
            particle = self._generator.generate().T      
        x = particle[:,0]
        y = particle[:,1]
        z = particle[:,2]
        extent = [[x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()]]
        grid_res = self.grid_res
        grid_res_sqr = grid_res**2

        # limits for random positioning of other particle
        x0 = (self.extent[0][0]-extent[0][1])
        x1 = (self.extent[0][1]-extent[0][0])
        y0 = (self.extent[1][0]-extent[1][1])
        y1 = (self.extent[1][1]-extent[1][0])        
        
        def z_separation(elem_1,elem_2):
            #elems = np.array()
            # faster not to vectorize this as:
            # ((elem_2[:2]-elem_1[:2])**2).sum()
            x_sep_sqr = (elem_1[0]-elem_2[0])**2 + (elem_1[1]-elem_2[1])**2 
            if x_sep_sqr >= grid_res_sqr:
                return np.inf
            else:
                #math.sqrt faster here
                return elem_1[2]-elem_2[2]-math.sqrt(grid_res_sqr-x_sep_sqr)

        site_found = False
        while not site_found:
            x_shift = x0+np.random.rand()*(x1-x0)
            y_shift = y0+np.random.rand()*(y1-y0)
            xs = x+x_shift
            ys = y+y_shift
                    
            overlapping_range = \
                [max(xs.min(),self.extent[0][0])-grid_res, 
                 min(xs.max(),self.extent[0][1])+grid_res, 
                 max(ys.min(),self.extent[1][0])-grid_res, 
                 min(ys.max(),self.extent[1][1])+grid_res]
                 
            if (overlapping_range[0] >= overlapping_range[1]) or \
                (overlapping_range[2] >= overlapping_range[3]): 
                
                #no overlap so impossible to connect -> stop
                if required:
                   continue
                else:
                   break   
        
            # elements from this particle that are candidates for connection
            X_filter = \
                (self.X[:,0] >= overlapping_range[0]) & \
                (self.X[:,0] < overlapping_range[1]) & \
                (self.X[:,1] >= overlapping_range[2]) & \
                (self.X[:,1] < overlapping_range[3])
            overlapping_X = self.X[X_filter,:]
            if not len(overlapping_X):
                if required:
                   continue
                else:
                   break
    
            elem_index = Index2D(elem_size=grid_res)            
            elem_index.insert(overlapping_X[:,:2],overlapping_X)
            
            # candidates from the connecting particle
            X_filter = \
                (xs >= overlapping_range[0]) & \
                (xs < overlapping_range[1]) & \
                (ys >= overlapping_range[2]) & \
                (ys < overlapping_range[3])
            overlapping_Xp = np.vstack((
                xs[X_filter],ys[X_filter],z[X_filter])).T
            
            min_z_sep = np.inf
            for elem in overlapping_Xp:
                candidates = elem_index.items_near(elem[:2], 
                    grid_res)
                min_z_sep = min(min_z_sep, min_z_separation(candidates, 
                    elem, grid_res_sqr))
                    
            site_found = not np.isinf(min_z_sep)            
            if not required:
                break            
   
        if site_found:
            zs = z+min_z_sep+pen_depth
            p_shift = np.vstack((xs,ys,zs)).T
            self.X = np.vstack((self.X,p_shift))
            if ident is not None:
                self.ident = np.hstack((self.ident, ident))
            else:
                ident = self.ident[0]
                self.ident = np.empty(self.X.shape[0], dtype=np.int32)
                self.ident.fill(ident)
            self.X -= self.X.mean(0)
            self.update_extent()    
            
        return site_found        
         
   
    def align(self):      
        #principal axes (unnormalized)
        (l, PA) = linalg.eigh(np.dot(self.X.T, self.X))
        l = np.array(l, dtype=float)
        PA = np.array(PA, dtype=float)
        ind = l.argsort()
        PA /= np.sqrt((PA**2).sum(0))
        PA = np.vstack((PA[:,ind[2]], PA[:,ind[1]], PA[:,ind[0]])).T
        self.X = np.dot(self.X,PA)
        self.update_extent()
         
         
    def rotate(self,rotator):
        self.X = self.X-self.X.mean(0)
        self.X = rotator.rotate(self.X.T).T
        self.update_extent()
         
    
    def visualize(self, bgcolor=(1,1,1), fgcolor=(.8,.8,.8)):
        color_list = [colors.colorConverter.to_rgb(c) for c in [
            "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", 
            "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", 
            "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
        ]]

        #local import as this can take a while
        from mayavi import mlab
        
        mlab.figure(bgcolor=bgcolor, fgcolor=fgcolor)
        i = 0
        for ident in xrange(self.ident.min(), self.ident.max()+1):
            X = self.X[self.ident==ident,:]
            if X.shape[0] > 0:
                mlab.points3d(X[:,0], X[:,1], X[:,2], 
                    color=color_list[i%len(color_list)],
                    mode="cube", scale_factor=self._generator.grid_res)
                i += 1
      
      
    def grid(self, res=None):
        if res==None:
          res = self.grid_res  

        def comp(x,y):
            if (x[0]==y[0]) and (x[1]==y[1]):
                return int(np.sign(x[2]-y[2]))
            elif (x[0]==y[0]):
                return int(np.sign(x[1]-y[1]))
            else:
                return int(np.sign(x[0]-y[0]))    

        def unique(a):
            la = list(a)            
            la.sort(cmp=comp)         
            a = np.array(la)         
            a_diff = abs(np.vstack((
                np.diff(a,axis=0),
                np.array([1.0,1.0,1.0]
            )))).sum(1)
            unique_filter = (a_diff != 0)
            return a[unique_filter,:], a[~unique_filter]
    
        Xc = (self.X/res).round().astype(int)
        (unique_X, duplicate_X) = unique(Xc)
        
        lu = set([tuple(p) for p in unique_X])
        ld = set([tuple(p) for p in duplicate_X])
        for d in ld:
            search_rad = 1
            site_found = False
            while not site_found:
                vacant = []
                for dx in xrange(-search_rad, search_rad+1):
                    for dy in xrange(-search_rad, search_rad+1):
                        for dz in xrange(-search_rad, search_rad+1):
                            c = (d[0]+dx, d[1]+dy, d[2]+dz)
                            if c not in lu:
                                vacant.append(c)
                if vacant:
                    lu.add(vacant[random.randint(len(vacant))])
                    site_found = True
                else:
                    search_rad += 1

        return np.array(sorted(lu, cmp=comp))



def spheres_overlap(X0, X1, r_sqr):
    return (X1[0]-X0[0])**2 + (X1[1]-X0[1])**2 + \
        (X1[2]-X0[2])**2 < r_sqr


class RimedAggregate(Aggregate):

    def add_rime_particles(self, N=1, pen_depth=120e-6):

        grid_res = self.grid_res
        grid_res_sqr = grid_res**2

        # limits for random positioning of rime particle
        x0 = (self.extent[0][0])
        x1 = (self.extent[0][1])
        y0 = (self.extent[1][0])
        y1 = (self.extent[1][1])

        use_indexing = (N > 1)

        if use_indexing:
            elem_index = Index2D(elem_size=grid_res)            
            elem_index.insert(self.X[:,:2],self.X)
            def find_overlapping(x,y):
                p_near = np.array(list(elem_index.items_near((x,y), grid_res)))
                if not p_near.shape[0]:
                    return p_near
                p_filter = ((p_near[:,:2]-[x,y])**2).sum(1) < grid_res_sqr
                return p_near[p_filter,:]
        else:
            def find_overlapping(x,y):
                X_filter = ((self.X[:,:2] - 
                    np.array([x,y]))**2).sum(1) < grid_res_sqr
                return self.X[X_filter,:]

        added_particles = np.empty((N, 3))

        for particle_num in xrange(N):
            site_found = False
            while not site_found:
                xs = x0+np.random.rand()*(x1-x0)
                ys = y0+np.random.rand()*(y1-y0)

                overlapping_range = [xs-grid_res, xs+grid_res,
                     ys-grid_res, ys+grid_res]

                overlapping_X = find_overlapping(xs, ys)
                if not overlapping_X.shape[0]:
                    continue                  
                
                X_order = overlapping_X[:,2].argsort()
                overlapping_X = overlapping_X[X_order,:]            
                last_ind = bisect_left(overlapping_X[:,2], 
                    overlapping_X[0,2]+pen_depth)
                last_search_ind = bisect_left(overlapping_X[:,2], 
                    overlapping_X[0,2]+pen_depth+grid_res)
                overlapping_X = overlapping_X[:last_search_ind+1,:]
                overlapping_z = overlapping_X[:,2]

                for i in xrange(last_ind-1, -1, -1):
                    d_sqr = (overlapping_X[i,0]-xs)**2 + (overlapping_X[i,1]-ys)**2
                    dz = math.sqrt(grid_res_sqr - d_sqr)
                    z_upper = overlapping_X[i,2] + dz
                    z_lower = overlapping_X[i,2] - dz                    
                    
                    for zc in [z_upper, z_lower]:
                        overlap = False
                        if (i==0) and (zc==z_lower):
                            break # automatically attach at the last site

                        j0 = bisect_left(overlapping_z, zc-grid_res)
                        j1 = bisect_left(overlapping_z, zc+grid_res)
                        
                        #search through possible overlapping spheres
                        for j in xrange(j0, j1):
                            if j == i:
                                continue
                            elif spheres_overlap(overlapping_X[j,:], (xs,ys,zc), 
                                grid_res_sqr):
                                # there is an overlapping sphere -> site unsuitable
                                overlap = True
                                break 

                        if not overlap:
                            break

                    if not overlap:
                        # this means we found a suitable site, so add the particle
                        added_particles[particle_num,:] = [xs, ys, zc]
                        site_found = True
                        self.extent[0][0] = min(self.extent[0][0], xs)
                        self.extent[0][1] = max(self.extent[0][1], xs)
                        self.extent[1][0] = min(self.extent[1][0], ys)
                        self.extent[1][1] = max(self.extent[1][1], ys)
                        self.extent[2][0] = min(self.extent[2][0], zc)
                        self.extent[2][1] = max(self.extent[2][1], zc)
                        if use_indexing:
                            elem_index.insert([[xs, ys]], [[xs, ys, zc]])
                        break

        self.X = np.vstack((self.X, added_particles))
        self.ident = np.hstack((self.ident, 
            -np.ones(added_particles.shape[0], dtype=np.int32)))
        self.X -= self.X.mean(0)
        self.update_extent()


         
         
class PseudoAggregate(Aggregate):
   def __init__(self, generator, sig=1.0):      
      self.sig = sig
      self.generator = generator
            
      self.X = self.generator.generate().T
      x = self.X[:,0]+stats.norm.rvs(scale=sig)
      y = self.X[:,1]+stats.norm.rvs(scale=sig)
      z = self.X[:,2]+stats.norm.rvs(scale=sig)
      self.extent = [[x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()]]      
         
                  
   def add_particle(self, particle=None, required=False):
      if particle == None:
         particle = self.generator.generate().T
      x = particle[:,0]+stats.norm.rvs(scale=self.sig)
      y = particle[:,1]+stats.norm.rvs(scale=self.sig)
      z = particle[:,2]+stats.norm.rvs(scale=self.sig)
      self.X = numpy.vstack((self.X, numpy.vstack((x,y,z)).T)) 
      x = self.X[:,0]
      y = self.X[:,1]
      z = self.X[:,2]
      self.extent = [[x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()]]
