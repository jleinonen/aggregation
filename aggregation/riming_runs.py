from riming import *

"""This is here for backward compatibility and to provide a command
line interface to rimed aggregate generation.
"""


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