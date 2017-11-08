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

import cPickle as pickle
import numpy as np
import crystal, aggregate, generator, rotator

def aspect_ratio():
    D_arr = np.exp(np.linspace(np.log(100e-6), np.log(3000e-6), 100))
    with open("../aggregation/dendrite_grid.dat") as f: 
        grid = pickle.load(f)
    rot = rotator.UniformRotator()
    grid_res = 40.0e-6

    def ar(D):
        cry = crystal.Dendrite(D, hex_grid=grid)
        gen = generator.MonodisperseGenerator(cry, rot, grid_res)
        agg = aggregate.Aggregate(gen)

        m = agg.X.mean(0)
        X_c = agg.X-m
        cov = np.dot(X_c.T, X_c)
        (l,v) = np.linalg.eigh(cov)
        size = np.sqrt(l/X_c.shape[0] + (1./(2*np.sqrt(3))*grid_res)**2)
        width = np.sqrt(0.5*(size[1]**2+size[2]**2))
        height = size[0]
        print D, size, width, height
        return height/width

    ratios = np.array([ar(D) for D in D_arr])
    return (D_arr, ratios)