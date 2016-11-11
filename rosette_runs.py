import gzip
import numpy as np
import aggregate, generator, rotator, crystal

def generate_rosette(D, min_grid_res=40e-6):
	grid_res = min(D/20, min_grid_res)
	rot = rotator.UniformRotator()	
   	cry = crystal.Rosette(D)
   	gen = generator.MonodisperseGenerator(cry, rot, grid_res)
   	agg = aggregate.Aggregate(gen)

   	return (agg, grid_res)

def generate_rosettes(D0=0.1e-3, D1=1e-2, N=1024):
	rosette_dir = "./rosettes/"
	diameters = np.exp(np.linspace(np.log(D0), np.log(D1), N))	
   	
   	meta_lines = []   	
	for (i,D) in enumerate(diameters):
		(agg, grid_res) = generate_rosette(D)
		fn = "rosette_{}.txt.gz".format(i+1024)
		area = agg.vertical_projected_area()
		N = agg.X.shape[0]
		with gzip.open(rosette_dir+"/"+fn, 'w') as f:
			np.savetxt(f, agg.grid(), fmt="%d")
		meta_lines.append("{:.6e} {:.6e} {:.6e} {:d} {}\n".format(D, grid_res, area, N, fn))

	with open("rosettes.txt", 'w') as f:
		for line in meta_lines:
			f.write(line)

