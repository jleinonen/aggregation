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
                
                
        
        
        
        
