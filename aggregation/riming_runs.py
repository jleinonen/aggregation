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