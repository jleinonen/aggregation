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

import argparse
try:
  import cPickle as pickle
except:
  import pickle
import gzip
import os
from numpy import random
import numpy as np
from scipy import stats
from . import aggregate, crystal, rotator, generator, mcs


rho_w = 1000.0
rho_i = 916.7


def get_N_rime_particles(agg, rot, riming_lwp, riming_eff=1.0, align=True,
    num_area_samples=10, debug=False):
    """Calculate the number of rime particles for a given LWP.

    Args:
        agg: The Aggregate object.
        rot: Rotator to apply before calculating the number of particles.
        riming_lwp: Liquid water path that the aggregate is assumed to fall
            through [kg/m^2].
        riming_eff: Riming efficiency (between 0 and 1, default 1); this 
            multiplies the riming_lwp parameter.
        align: If True, the aggregate is aligned before calculating the
            number of particles.
        num_area_samples: How many times the area should be sampled;
            the effective area is the mean of num_area_samples tests.
        debug: If True, print debug information about the calculation.

    Returns:
        Tuple (N_particles, area), where N_particles is the number of
        rime particles needed, and area is the averaged projected area [m^2].
    """

    area_list = []
    for i in xrange(num_area_samples):
        area_list.append(agg.vertical_projected_area())
        if align:
            agg.align()
        agg.rotate(rot)
    area = np.mean(area_list)

    vol = riming_lwp * area / rho_w    
    N_particles = int(round(riming_eff*vol/agg.grid_res**3))
    if debug:
        print(riming_lwp, area, vol, N_particles)
    return (N_particles, area)


def lwp_from_N(agg, N, area):
    """The liquid water path corresponding to a number of rime particles.
    
    Args:
        N: The number of particles.
        area: The particle effective area.

    Returns:
        The liquid water path [kg/m^2].
    """

    vol = N*agg.grid_res**3
    return vol * rho_w / area


def generate_rime(agg, rot, riming_lwp, riming_eff=1.0, align=True, 
    pen_depth=120e-6, lwp_div=10.0, compact_dist=0., iter=False):
    """Generate rime on an aggregate.
    
    Args:
        agg: The Aggregate object.
        rot: Rotator to apply before calculating the number of particles.
        riming_lwp: Liquid water path that the aggregate is assumed to fall
            through [kg/m^2].
        riming_eff: Riming efficiency (between 0 and 1, default 1); this 
            multiplies the riming_lwp parameter.
        align: If True, the aggregate is aligned before calculating the
            number of particles.
        pen_depth: The distance that rime particles are allowed to penetrate
            into the particle.
        lwp_div: The inverse fraction of riming_lwp that is delivered in a
            single orientation. E.g. if this is 10, the LWP is added in 10
            approximately equal parts and the aggregate is realigned and
            rotated between each addition.
        iter: If True, this function will return an iterator that gives the
            different stages of the aggregate growth. This can be useful
            if one wants to follow the riming process.

    Returns:
        A generator if iter=True, None otherwise.
    """

    def gen():    
        remaining_lwp = riming_lwp

        while remaining_lwp > 0:
            (N_particles, area) = get_N_rime_particles(agg, rot, 
                min(riming_lwp/lwp_div, remaining_lwp), riming_eff, align=align)
            N_to_add = max(N_particles, 10)
            agg.add_rime_particles(N=N_to_add, pen_depth=pen_depth, 
                compact_dist=compact_dist)
            remaining_lwp -= lwp_from_N(agg, N_to_add, area)
            if align:
                agg.align()
            agg.rotate(rot)
            yield agg

    if iter:
        return gen()
    else:
        for i in gen():
            pass


def generate_rimed_aggregate(*args, **kwargs):
    """Generate a rimed aggregate particle.
    
    Args:
        monomer_generator: The Generator object used to make the ice 
            crystals.
        N: Number of ice crystals to aggregate.
        align: If True, the aggregate is kept horizontally aligned between
            iterations.
        riming_lwp: Liquid water path that the aggregate is assumed to fall
            through [kg/m^2].
        riming_eff: Riming efficiency (between 0 and 1, default 1); this 
            multiplies the riming_lwp parameter.
        riming_mode: "simultaneous" if the aggregation and riming occur
            simultaneously, or "subsequent" if all aggregation is done first,
            followed by the riming.
        pen_depth: The distance that rime particles are allowed to penetrate
            into the particle.
        seed: Random seed. If not None, this seed is used to initialize the
            random generator.
        lwp_div: See the documentation for generate_rime for details.
        debug: If True, print additional debugging information.
        iter: If True, this function will return an iterator that gives the
            different stages of the aggregation and riming.

    Returns:
        If iter=True, a generator giving the different stages of aggregate
        formation and riming; each iteration returns a list of the particles
        at that stage. If iter=False, returns the final aggregate particle as
        an Aggregate instance (this is equivalent to the final iteration when
        iter=True).
    """

    if "iter" in kwargs:
        iter = kwargs["iter"]
        del kwargs["iter"]
    else:
        iter = False

    if iter:
        def generator():
            for aggs in generate_rimed_aggregate_iter(*args, **kwargs):
                yield aggs
        return generator()
    else:
        aggs = None
        for aggs in generate_rimed_aggregate_iter(*args, **kwargs):
            pass
        return aggs[0]


def generate_rimed_aggregate_iter(monomer_generator, N=5, align=True,
    riming_lwp=0.0, riming_eff=1.0, riming_mode="simultaneous", 
    rime_pen_depth=120e-6, seed=None, lwp_div=10, compact_dist=0.,
    debug=False):
    """Generate a rimed aggregate particle.

    Refer to the generate_rimed_aggregate function for details.
    """

    random.seed(seed)

    align_rot = rotator.PartialAligningRotator(exp_sig_deg=40, random_flip=True)
    uniform_rot = rotator.UniformRotator()

    agg = [monomer_generator(ident=i) for i in xrange(N)]
    yield agg

    while len(agg) > 1:
        r = np.array([((a.extent[0][1]-a.extent[0][0])+
            (a.extent[1][1]-a.extent[1][0]))/4.0 for a in agg])
        m_r = np.sqrt(np.array([a.X.shape[0] for a in agg])/r)
        r_mat = (np.tile(r,(len(agg),1)).T+r)**2                
        mr_mat = abs(np.tile(m_r,(len(agg),1)).T - m_r)
        p_mat = r_mat * mr_mat
        p_mat /= p_mat.max()
        collision = False
        while not collision:
            
            i = random.randint(len(agg))
            j = i
            while j == i:
                j = random.randint(len(agg))
            rnd = random.rand()
            if rnd < p_mat[i][j]:
                if debug:
                    print(i, j)
                agg_top = agg[i] if (m_r[i] > m_r[j]) else agg[j]
                agg_btm = agg[i] if (m_r[i] <= m_r[j]) else agg[j]
                agg_btm.rotate(uniform_rot)
                collision = agg_top.add_particle(particle=agg_btm.X,ident=agg_btm.ident,
                    required=True,pen_depth=80e-6)
                if collision:
                    if align:
                        agg_top.align()
                        agg_top.rotate(align_rot)
                    else:
                        agg_top.rotate(uniform_rot)
                    agg.pop(i if (m_r[i] <= m_r[j]) else j)

        if riming_mode == "simultaneous":
            for a in agg:                
                generate_rime(a, align_rot if align else uniform_rot,
                    riming_lwp/float(N-1), riming_eff=riming_eff, 
                    pen_depth=rime_pen_depth, lwp_div=lwp_div,
                    compact_dist=compact_dist)

        if len(agg) > 1:
            yield agg

    if (riming_mode == "subsequent") or (N==1):
        for a in generate_rime(agg[0], align_rot if align else uniform_rot, 
            riming_lwp, riming_eff=riming_eff, pen_depth=rime_pen_depth, 
            lwp_div=lwp_div, iter=True, compact_dist=compact_dist):

            yield [a]

    if align:
        agg[0].align()
        agg[0].rotate(align_rot)
    agg[0].rotate(rotator.HorizontalRotator())

    yield agg


def gen_polydisperse_monomer(monomers=[], ratios=[]):
    """ Make a monomer crystal generator that picks from multiple distributions
        of different monomers, each with its own crystal type and PSD.

    Args:
        monomers: list of dictionaries. Each dictionary contains the list of
            arguments of for the single monomer generator gen_monomer for each
            monomer type
        ratios: list of floats describing the relative contributions of each
            monomer type to the overall population. The function first picks the
            monomer type according to the distribution defined by this list of
            ratios, than it picks a single monomer according to its own PSD.

    Returns:
        gen: a generator of monomers that takes into account the presence of
            multiple distributions of monomers allowing to generate aggregates
            of monomers of different shapes or coming from bimodal distributions
            of monomers
    """
    
    if len(monomers) != len(ratios):
        raise AttributeError('The length of the list of monomers must  match'+ \
                             'the length of the list of ratios')
    
    if sum(ratios) != 1.0:
        print('Warning! Distro ratios do not sum up to 1.0 ... normalizing')

    genlist = [gen_monomer(**i) for i in monomers]
    cumsum = np.cumsum(ratios)

    def polygen(ident=0):
        rnd = np.random.uniform()
        i = np.arange(len(genlist))[cumsum > rnd][0]
        return genlist[i](ident)

    return polygen


def gen_monomer(psd="monodisperse", size=1e-3, min_size=0.1e-3, 
    max_size=20e-3, mono_type="dendrite", grid_res=0.02e-3, 
    rimed=False, debug=False):
    """Make a monomer crystal generator.

    Args:
        psd: "monodisperse" or "exponential".
        size: If psd="monodisperse", this is the diameter of the ice
            crystals. If psd="exponential", this is the inverse of the
            slope parameter (usually denoted as lambda).
        min_size, max_size: Minimum and maximum allowed size for generated
            crystals.
        mono_type: The type of crystals used. The possible values are
            "dendrite", "plate", "needle", "rosette", "bullet", "column",
            "spheroid".
        grid_res: The volume element size.
        rimed: True if a rimed aggregate should be generated, False
            otherwise.
        debug: If True, debug information will be printed.
    """
        
    def make_cry(D):
        if mono_type=="dendrite":
            current_dir = os.path.dirname(os.path.realpath(__file__))
            grid = pickle.load(file(current_dir+"/dendrite_grid.dat"))
            cry = crystal.Dendrite(D, hex_grid=grid)
        elif mono_type=="plate":
            cry = crystal.Plate(D)            
        elif mono_type=="needle":
            cry = crystal.Needle(D)
        elif mono_type=="rosette":
            cry = crystal.Rosette(D)
        elif mono_type=="bullet":
            cry = crystal.Bullet(D)
        elif mono_type=="column":
            cry = crystal.Column(D)
        elif mono_type=="spheroid":
            cry = crystal.Spheroid(D,1.0)
        return cry
                
    rot = rotator.UniformRotator()            
    
    def gen(ident=0):
        if psd=="monodisperse":
            D = size
        elif psd=="exponential":
            psd_f = stats.expon(scale=size)
            D=max_size+1
            while (D<min_size) or (D>max_size):
                D = psd_f.rvs()
        
        cry = make_cry(D)
        if debug:
            print(D, mono_type)
        
        gen = generator.MonodisperseGenerator(cry, rot, grid_res)
        if rimed:
            agg = aggregate.RimedAggregate(gen, ident=ident)
        else:
            agg = aggregate.Aggregate(gen, ident=ident)
        return agg
    
    return gen


def visualize_crystal(mono_type):
    gen = gen_monomer(mono_type=mono_type, size=2e-3, grid_res=40e-6)
    cry = gen()
    cry.align()
    cry.visualize(bgcolor=(1,1,1))
