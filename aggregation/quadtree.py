from itertools import izip

class TreeNode:   
   #bounds: (left, right, bottom, top)
   def __init__(self, levels, bounds, tree):      
      self.bounds = bounds
      self.levels = levels
      self.items = []
      self.has_items = False
      if levels > 1:
         b00 = (bounds[0], (bounds[0]+bounds[1])/2.0, (bounds[2]+bounds[3])/2.0, bounds[3])
         b01 = ((bounds[0]+bounds[1])/2.0, bounds[1], (bounds[2]+bounds[3])/2.0, bounds[3])
         b10 = (bounds[0], (bounds[0]+bounds[1])/2.0, bounds[2], (bounds[2]+bounds[3])/2.0)
         b11 = ((bounds[0]+bounds[1])/2.0, bounds[1], bounds[2], (bounds[2]+bounds[3])/2.0)
         self.children = [TreeNode(levels-1,b00,tree),TreeNode(levels-1,b01,tree),\
                           TreeNode(levels-1,b10,tree),TreeNode(levels-1,b11,tree)]
      else:
         self.children = None
         
      
   #coords: (x, y)
   def insert(self, item, coords, diam):
      self.has_items = True
      if self.children == None:
         self.items.append(item)
      else:
         for child in self.children:
            if child.is_inside(coords, diam):         
               child.insert(item, coords, diam)       
   
   
   def is_inside(self, coords, diam):
      (x,y) = coords
      (left,right,bottom,top) = self.bounds
      return (x>=left-diam) and (x<right+diam) and (y>=bottom-diam) and (y<top+diam)   
   
   
   def get_items(self):
      if self.has_items:
         if self.children == None:
            return self.items
         else:
            return self.children[0].get_items() + \
                   self.children[1].get_items() + \
                   self.children[2].get_items() + \
                   self.children[3].get_items()
      else:
         return []
   
   
   def overlap(self, other):         
      if self.has_items and other.has_items:
         if self.children == None:
            return [(self.items,other.items)]
         else:
            return self.children[0].overlap(other.children[0]) + \
                   self.children[1].overlap(other.children[1]) + \
                   self.children[2].overlap(other.children[2]) + \
                   self.children[3].overlap(other.children[3])
      else:
         return []
   
   
         
      
class Quadtree:
   def __init__(self, levels, bounds):
      if (bounds[0] >= bounds[1]) or (bounds[2] >= bounds[3]):
         raise ValueError("Required: top > bottom and right > left.")
      self.levels = levels
      self.bounds = bounds
      self.root = TreeNode(levels, bounds, self)
      
      
   def insert(self, item, coords, rad):
      #if not self.root.is_inside(coords):
      #   raise ValueError("Coordinates not within bounds.")
      self.root.insert(item, coords, rad)  
             
    
   def get_items(self):   
      return self.root.get_items()
             
      
   def overlap(self, other):
      if (self.root.bounds != other.root.bounds) or (self.root.levels != other.root.levels):
         raise ValueError("Both trees must have the same bounds and levels.")
      return self.root.overlap(other.root)      
      
