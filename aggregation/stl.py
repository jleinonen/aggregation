import numpy as np


def interface_square(x,y,z,axis,direction):
    axis_increment = (1 if direction==1 else 0)

    ranges = [(0,1),(0,1),(0,1)]
    ranges[axis] = (axis_increment,)
    (dx_range, dy_range, dz_range) = ranges

    square = []
    for dx in dx_range:
        for dy in dy_range:
            for dz in dz_range:
                square.append([x+dx,y+dy,z+dz])

    return square


def normal_vector(axis,direction):
    n = [0,0,0]
    n[axis] = direction

    return n


def interface_squares_from_grid(X):
    # shoft coordinates to non-negative to comply with STL standard
    X = X - X.min()

    # find the limits of the grid
    (x_min, y_min, z_min) = (0,0,0)
    (x_max, y_max, z_max) = X.max(axis=0)+1

    # use the set as an index of the points
    points = set(tuple(x) for x in X)
    squares = []
    normals = []

    for x in range(x_min,x_max):
        for y in range(y_min,y_max):
            for z in range(z_min,z_max):
                if (x,y,z) not in points:
                    continue
                # the rest of the loop assumes we're in a non-empty point

                
                for axis in [0,1,2]:
                    for direction in [-1,1]:
                        neighbor = [x,y,z]
                        neighbor[axis] += direction
                        if tuple(neighbor) not in points:
                            # if the neighbor is not in "points"
                            # we have an interface
                            squares.append(
                                interface_square(x,y,z,axis,direction))
                            normals.append(normal_vector(axis,direction))

    squares = np.array(squares)

    return (squares, np.array(normals))


def squares_to_triangles(squares, normals):
    return (
        np.concatenate([squares[:,:-1,:],squares[:,1:,:]]),
        np.concatenate([normals,normals])
    )


def rotate_triangle(triangle, normal):
    # Ensure that the triangle is oriented correctly according to the
    # STL specification
    v = triangle[0,:]-triangle[1,:]
    w = triangle[2,:]-triangle[1,:]
    cross = np.cross(v,w)
    dot = cross.dot(normal)
    if dot > 0:
        triangle = triangle[::-1,:]
    return triangle


def triangles_to_stl(triangles, normals, stl_fn, mode='ascii'):

    if mode == 'ascii':
        header = 'solid snowflake\n'
        write_mode = 'w'

    def encode_normal(normal):
        if mode == 'ascii':
            packed_normal = 'facet normal {:d} {:d} {:d}\n'.format(*normal)
        return packed_normal

    def encode_vertex(vertex):
        if mode == 'ascii':
            packed_vertex = '        vertex {:d} {:d} {:d}\n'.format(*vertex)
        return packed_vertex

    def write_triangle_header(f, normal):
        packed_normal = encode_normal(normal)
        f.write(packed_normal)
        if mode=='ascii':
            f.write('    outer loop\n')

    def write_triangle_footer(f):
        if mode=='ascii':
            f.write('    endloop\nendfacet\n')

    with open(stl_fn, write_mode) as f:
        f.write(header)

        for (triangle,normal) in zip(triangles,normals):
            triangle = rotate_triangle(triangle,normal)
            write_triangle_header(f,normal)
            for vertex in triangle:
                packed_vertex = encode_vertex(vertex)
                f.write(packed_vertex)
            write_triangle_footer(f)

        if mode=='ascii':
            f.write('endsolid snowflake\n')


def snowflake_grid_to_stl(X, stl_fn, mode='ascii'):
    """Produce an STL file from a grid model.

    Produces an STL format 3D model file from a gridded snowflake model. The
    STL file will use the integer coordinates which can be multiplied by
    the grid resolution to obtain the physical coordinates in meters.

    Params:
        X: A (N,3) shaped array of integer spaced coordinates of the volume
            elements. Can be obtained with Aggregate.grid() (or loaded from
            a previously saved file).
        stl_fn: The output STL file.
        mode: STL file type. Currently only 'ascii' is supported. Support for
            binary STL files may be added in the future.
    """

    (squares, normals) = interface_squares_from_grid(X)
    (triangles, normals) = squares_to_triangles(squares, normals)
    triangles_to_stl(triangles, normals, stl_fn)
