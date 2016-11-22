import argparse
import cPickle as pickle
import gzip
import json
import os
from numpy import array, random
import numpy as np
from scipy import stats
import aggregate, crystal, rotator, generator, mcs


rho_w = 1000.0
rho_i = 916.7


def get_N_rime_particles(agg, rot, riming_lwp, riming_eff, align=True,
    num_area_samples=10, debug=False):

    area_list = []
    for i in xrange(num_area_samples):
        area_list.append(agg.vertical_projected_area())
        if align:
            agg.align()
        agg.rotate(rot)
    area = np.mean(area_list)

    vol = riming_lwp * area / rho_w    
    N_particles = int(round(vol/agg.grid_res**3))
    if debug:
        print riming_lwp, area, vol, N_particles
    return (N_particles, area)


def lwp_from_N(agg, N, area):
    vol = N*agg.grid_res**3
    return vol * rho_w / area


def generate_rime(agg, rot, riming_lwp, riming_eff=1.0, align=True, pen_depth=120e-6,
    lwp_div=10.0, iter=False):

    def gen():    
        remaining_lwp = riming_lwp

        while remaining_lwp > 0:
            (N_particles, area) = get_N_rime_particles(agg, rot, 
                min(riming_lwp/lwp_div, remaining_lwp), riming_eff, align=align)
            N_to_add = max(N_particles, 10)
            agg.add_rime_particles(N=N_to_add, pen_depth=pen_depth)
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
    rime_pen_depth=120e-6, seed=None, lwp_div=10, debug=False):

    random.seed(seed)

    align_rot = rotator.PartialAligningRotator(exp_sig_deg=40, random_flip=True)
    uniform_rot = rotator.UniformRotator()

    agg = [monomer_generator(ident=i) for i in xrange(N)]
    yield agg

    while len(agg) > 1:
        r = array([((a.extent[0][1]-a.extent[0][0])+(a.extent[1][1]-a.extent[1][0]))/4.0 for a in agg])
        m_r = np.sqrt(array([a.X.shape[0] for a in agg])/r)
        r_mat = (np.tile(r,(len(agg),1)).T+r)**2                
        mr_mat = abs(np.tile(m_r,(len(agg),1)).T - m_r)
        p_mat = r_mat * mr_mat
        p_mat /= p_mat.max()
        collision = False
        while not collision:
            
            i = random.randint(len(agg))
            j = random.randint(len(agg))
            rnd = random.rand()
            if rnd < p_mat[i][j]:
                if debug:
                    print i, j
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
                    pen_depth=rime_pen_depth, lwp_div=lwp_div)

        if len(agg) > 1:
            yield agg

    if (riming_mode == "subsequent") or (N==1):
        for a in generate_rime(agg[0], align_rot if align else uniform_rot, 
            riming_lwp, riming_eff=riming_eff, pen_depth=rime_pen_depth, 
            lwp_div=lwp_div, iter=True):

            yield [a]

    if align:
        agg[0].align()
        agg[0].rotate(align_rot)
    agg[0].rotate(rotator.HorizontalRotator())

    yield agg


def gen_monomer(psd="monodisperse", size=1.0, min_size=1e-3, max_size=10,
    mono_type="dendrite", grid_res=0.02e-3, rimed=False, debug=False):
        
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
            print D
        
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--psd', required=True)
    parser.add_argument('--mono_type', required=True)
    parser.add_argument('--mono_size', type=float, required=True)
    parser.add_argument('--mono_min_size', type=float, default=None)
    parser.add_argument('--mono_max_size', type=float, default=None)
    parser.add_argument('--num_monos', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--grid_res', type=float, required=True)
    parser.add_argument('--riming_mode', required=True)
    parser.add_argument('--riming_lwp', type=float, required=True)
    parser.add_argument('--riming_eff', type=float, required=True)
    parser.add_argument('--rime_pen_depth', type=float, required=True)
    args = parser.parse_args()

    mono_generator = gen_monomer(psd=args.psd, size=args.mono_size, 
        min_size=args.mono_min_size, max_size=args.mono_max_size,
        mono_type=args.mono_type, grid_res=args.grid_res, rimed=True)
        
    agg = generate_rimed_aggregate(mono_generator, N=args.num_monos, align=True,
        riming_lwp=args.riming_lwp, riming_eff=args.riming_eff, 
        riming_mode=args.riming_mode, rime_pen_depth=args.rime_pen_depth)

    D_max = (2*mcs.minimum_covering_sphere(agg.X)[1])+args.grid_res
    r_g = np.sqrt(((agg.X-agg.X.mean(0))**2).sum(1).mean())

    meta = {"psd": args.psd, "mono_type": args.mono_type, 
        "mono_size": args.mono_size, "mono_min_size": args.mono_min_size,
        "mono_max_size": args.mono_max_size, "num_monos": args.num_monos,
        "grid_res": args.grid_res, "file_name": args.output,
        "riming_mode": args.riming_mode, "riming_lwp": args.riming_lwp, 
        "riming_eff": args.riming_eff, "extent": agg.extent,
        "mass": agg.X.shape[0]*agg.grid_res**3*rho_i,
        "N_elem": agg.X.shape[0], "N_rimed": (agg.ident==-1).sum(),
        "max_diam": D_max, "rad_gyration": r_g
        }
    with gzip.open(args.output, 'w') as f:
        np.savetxt(f, agg.grid(), fmt="%d")
    json.dump(meta, file(args.output+".meta", 'w'))
