from itertools import izip, chain, product
import numpy as np


class Index2D(object):
    def __init__(self, elem_size=1):
        self._elem_size = float(elem_size)
        self._grid = {}
        
        
    def insert(self, coordinates, objects=None):
        if objects is None:
            objects = coordinates
        X = np.array(coordinates)/self._elem_size
        X_i = X.astype(np.int32)
        for ((x,y),(x_i,y_i),obj) in izip(X,X_i,objects):
            try:
                self._grid[(x_i,y_i)].append(((x,y),obj))
            except KeyError:
                self._grid[(x_i,y_i)] = [((x,y),obj)]
                
    
    def _items_in_cell(self, x_i, y_i):
        try:
            return self._grid[(x_i,y_i)]
        except KeyError:
            return []

        
    def items_near(self, p, search_rad=1):
        """Return all indexed items within search_rad from p
        """
        
        p = np.array(p)/self._elem_size        
        search_rad = search_rad/self._elem_size
        (px, py) = p        
    
        items = []
        
        #return nearby cells
        cell_x = xrange(int(px-search_rad), int(px+search_rad)+1)
        cell_y = xrange(int(py-search_rad), int(py+search_rad)+1)
        
        for (x_i, y_i) in product(cell_x, cell_y):   
            items.append(item[1] for item in self._items_in_cell(x_i, y_i))
                
        return chain(*items)


class Index3D(object):
    def __init__(self, elem_size=1):
        self._elem_size = float(elem_size)
        self._grid = {}

    def size(self):
        return sum(len(self._grid[c]) for c in self._grid)
        
        
    def insert(self, coordinates):
        X = np.array(coordinates)/self._elem_size
        X_i = X.astype(np.int32)
        for ((x,y,z),(x_i,y_i,z_i)) in izip(X,X_i):
            try:
                self._grid[(x_i,y_i,z_i)].append((x,y,z))
            except KeyError:
                self._grid[(x_i,y_i,z_i)] = [(x,y,z)]


    def remove(self, coordinates):
        X = np.array(coordinates)/self._elem_size
        X_i = X.astype(np.int32)
        for ((x,y,z),(x_i,y_i,z_i)) in izip(X,X_i):
            try:
                ind = self._grid[(x_i,y_i,z_i)].index((x,y,z))
            except ValueError:
                print (x,y,z)
                print self._grid[(x_i,y_i,z_i)]
                raise 
            self._grid[(x_i,y_i,z_i)].pop(ind)
            if len(self._grid[(x_i,y_i,z_i)]) == 0:
                del self._grid[(x_i,y_i,z_i)]

    
    def _items_in_cell(self, x_i, y_i, z_i):
        try:
            return [(x*self._elem_size,y*self._elem_size,z*self._elem_size) 
                for (x,y,z) in self._grid[(x_i,y_i,z_i)]]
        except KeyError:
            return []

        
    def items_near(self, p, search_rad=1):
        """Return all indexed items within search_rad from p
        """
        
        p = np.array(p)/self._elem_size        
        search_rad = search_rad/self._elem_size
        (px, py, pz) = p        
    
        items = []
        
        #return nearby cells
        cell_x = xrange(int(px-search_rad), int(px+search_rad)+1)
        cell_y = xrange(int(py-search_rad), int(py+search_rad)+1)
        cell_z = xrange(int(pz-search_rad), int(pz+search_rad)+1)
        
        for (x_i, y_i, z_i) in product(cell_x, cell_y, cell_z):   
            items.append(item for item in self._items_in_cell(x_i, y_i, z_i))
                
        return chain(*items)
