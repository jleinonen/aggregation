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

try:
  from itertools import izip as zip
except:
  pass
import numpy as np
import math


def mcsc(points, candidate_indices):
    candidate_count = len(candidate_indices)
    candidate_center = np.array([0.0, 0.0, 0.0])
    if candidate_count == 1:
        candidate_center = points[candidate_indices[0],:]
    elif candidate_count == 2:
        q0 = points[candidate_indices[0],:]
        q1 = points[candidate_indices[1],:]
        #l0 = l1 = 0.5
        #candidate_center then is midpoint formula
        candidate_center = (q0 + q1)/2.0
    elif candidate_count == 3:
        q0 = points[candidate_indices[0],:]
        q1 = points[candidate_indices[1],:]
        q2 = points[candidate_indices[2],:]        
        a00 = np.dot(q0 - q2, q0 - q2)
        a01 = np.dot(q0 - q2, q1 - q2)
        a10 = np.dot(q1 - q2, q0 - q2)
        a11 = np.dot(q1 - q2, q1 - q2)
  
        A = np.array([[a00,a01],[a10,a11]])
        b = np.array([a00/2.0,a11/2.0])
            
        try:
            l = np.linalg.solve(A.T, b)
  
            l_sum = l[0] + l[1]
            l2 = 1.0 - l_sum
        
            l_list = [l[0], l[1], l2]
  
            drop_index = None
            minimum = 0
  
            for (index, number) in enumerate(l_list):
                if number < minimum:
                    drop_index = index
                    minimum = number
     
            if drop_index is not None:
                candidate_indices.pop(drop_index)
                (candidate_center, candidate_indices) = mcsc(points, candidate_indices)
            else:
                candidate_center = l[0]*q0 + l[1]*q1 + l2*q2
        except np.linalg.LinAlgError:
            candidate_indices.pop()
            (candidate_center, candidate_indices) = mcsc(points, candidate_indices)
 
    elif candidate_count == 4:
        q0 = points[candidate_indices[0],:]
        q1 = points[candidate_indices[1],:]
        q2 = points[candidate_indices[2],:]
        q3 = points[candidate_indices[3],:]
        a00 = (q0 - q3).dot(q0 - q3)
        a01 = (q0 - q3).dot(q1 - q3)
        a02 = (q0 - q3).dot(q2 - q3)
        a10 = (q1 - q3).dot(q0 - q3)
        a11 = (q1 - q3).dot(q1 - q3)
        a12 = (q1 - q3).dot(q2 - q3)
        a20 = (q2 - q3).dot(q0 - q3)
        a21 = (q2 - q3).dot(q1 - q3)
        a22 = (q2 - q3).dot(q2 - q3)
  
        A = np.array([[a00, a01, a02],[a10, a11, a12],[a20, a21, a22]])
        b = np.array((a00/2.0, a11/2.0, a22/2.0))
            
        try:            
            l = np.linalg.solve(A.T, b)
  
            l_sum = l[0] + l[1] + l[2]
            l3 = 1.0 - l_sum
        
            l_list = [l[0], l[1], l[2], l3]
  
            drop_index = None
            minimum = 0
  
            for (index, number) in enumerate(l_list):
                if number < minimum:
                    drop_index = index
                    minimum = number
     
            if drop_index is not None:
                candidate_indices.pop(drop_index)
                (candidate_center, candidate_indices) = mcsc(points, candidate_indices)
            else:
                candidate_center = l[0]*q0 + l[1]*q1 + l[2]*q2 + l3*q3
        except np.linalg.LinAlgError: # singular matrix
            candidate_indices.pop()
            (candidate_center, candidate_indices) = mcsc(points, candidate_indices)

        
    return (candidate_center, candidate_indices)


def find_next_candidate(points, center, candidate_indices):
    """Reduce covering sphere along line p*(t - center)
    New candidates have p values between 0 exclusive and 1 exclusive
    Candidate with smallest value is chosen for next covering sphere
    If minimum_covering sphere_candidate returns 4 points then minimum found
    If no new candidate points are found then minimum is found
    If new_center == center minimum is found
    """
    t = 0.0
    (t, candidate_indices) = mcsc(points, candidate_indices)
    if len(candidate_indices) == 4:
        return (True, t, candidate_indices)
    
    p = np.ones(points.shape[0])
    p0 = points[candidate_indices[0],:]        

    non_candidate_indices = np.setdiff1d(np.arange(points.shape[0]), candidate_indices)
    non_candidate_points = points[non_candidate_indices, :]  
    
    d = -(non_candidate_points-p0).dot(t-center)
    d_pos = d > 0    
    if d_pos.any():
        pos_points = non_candidate_points[d_pos]
        pp = -(((pos_points+p0)/2.0 - center) * \
            ((pos_points-p0)/d[d_pos,None])).sum(1)
        pn = p[non_candidate_indices]
        pn[d_pos] = pp
        p[non_candidate_indices] = pn   
    
    minimum = p[p>0].min()
    min_index = np.where(p==minimum)[0][-1]

    if minimum == 1:
        return (True, t, candidate_indices)
 
    new_center = center + p[min_index]*(t - center)
 
    candidate_indices.insert(0, min_index) 
 
    if (new_center == center).all():
        return (True, new_center, candidate_indices)

    return (False, new_center, candidate_indices)


def minimum_covering_sphere(points):
    """Minimum covering sphere.

    Based on http://www.mel.nist.gov/msidlibrary/doc/hopp95.pdf

    Args:
        points: (N,3) array of points.

    Returns:
        A tuple (center, radius) where center is a (3,) array with the
        coordinates of the center of the 
    """
    points = np.asarray(points).astype(np.float64)
    point_0 = points[0,:]
    new_center = point_0
    
    max_d = 0.0
    point_1_index = 0
    
    dd = points-point_0
    d = np.sqrt((dd*dd).sum(1))
    point_1_index = d.argmax()
    max_d = d[point_1_index]
   
    candidate_indices = [point_1_index]
    finished = False
    while not finished:
        center = new_center
        (finished, new_center, candidate_indices) = find_next_candidate(
            points, center, candidate_indices)
        #print(candidate_indices)
    diff = points[candidate_indices[0],:] - new_center
    radius = math.sqrt(np.dot(diff,diff))
    return (new_center, radius)

