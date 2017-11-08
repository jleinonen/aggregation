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

from itertools import izip, chain, product
import numpy as np


class Index2D(object):
    """Index objects by their 2D coordinates.

    This class can be used for fast spatial searching from a coordinate 
    dataset.

    Constructor args:
        elem_size: Spacing of the 2D grid used for indexing.
    """

    def __init__(self, elem_size=1):
        self._elem_size = float(elem_size)
        self._grid = {} # this holds the index
        
        
    def insert(self, coordinates, objects=None):
        """Insert objects into the spatial index.

        Args:
            coordinates: A (N,2) array with the spatial coordinates of the
                objects.
            objects: An iterator is N indexed objects. If None (default), the 
                (2,) coordinate arrays themselves are indexed.
        """

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
        """Return all indexed items within a given distance from a point.

        Some items that are further than search_rad from p may also be
        returned.

        Args:
            p: The reference point.
            search_rad: The search radius.

        Returns:
            An iterator containing the indexed items near p.
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
    """Index 3D coordinates.

    This class can be used for fast spatial searching from a coordinate 
    dataset.

    Constructor args:
        elem_size: Spacing of the 3D grid used for indexing.
    """

    def __init__(self, elem_size=1):
        self._elem_size = float(elem_size)
        self._grid = {}


    def size(self):
        """The size of the index.

        Returns:
            The total number of indexed coordinates.
        """
        return sum(len(self._grid[c]) for c in self._grid)
        
        
    def insert(self, coordinates):
        """Insert coordinates into the spatial index.

        Args:
            coordinates: A (N,3) array with the spatial coordinates to
                be indexed.
        """

        X = np.array(coordinates)/self._elem_size
        X_i = X.astype(np.int32)
        for ((x,y,z),(x_i,y_i,z_i)) in izip(X,X_i):
            try:
                self._grid[(x_i,y_i,z_i)].append((x,y,z))
            except KeyError:
                self._grid[(x_i,y_i,z_i)] = [(x,y,z)]


    def remove(self, coordinates):
        """Remove coordinates the spatial index.

        Args:
            coordinates: A (N,3) array with the spatial coordinates to be
                removed from the index.
        """

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
        """Return all indexed items within a given distance from a point.

        Some items that are further than search_rad from p may also be
        returned.

        Args:
            p: The reference point.
            search_rad: The search radius.

        Returns:
            An iterator containing the indexed items near p.
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
