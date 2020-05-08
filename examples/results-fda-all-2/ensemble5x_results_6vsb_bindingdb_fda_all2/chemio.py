#!/usr/bin/python


import numpy
import struct
import sys


############################################################


def read_vox(name):
    with open(name, 'rb') as f:
        # Read channel count and channel names.
        c = struct.unpack('i',f.read(4))[0]
        t = f.read(c)
        # Read window position.
        x = struct.unpack('i',f.read(4))[0]
        y = struct.unpack('i',f.read(4))[0]
        z = struct.unpack('i',f.read(4))[0]
        # Read window size.
        w = struct.unpack('i',f.read(4))[0]
        h = struct.unpack('i',f.read(4))[0]
        d = struct.unpack('i',f.read(4))[0]
        # Read voxels.
        b = f.read(c*w*h*d*4)
        voxels = numpy.frombuffer(b, dtype=numpy.float32)
        voxels = numpy.reshape(voxels, (c,w,h,d), order='F')
        return c, t, x, y, z, voxels


def read_lig(name):
    with open(name, 'r') as f:
        # Read number of atoms.
        n = int(next(f).strip())
        # Read atomic charges.
        a = [int(x) for x in next(f).split()]
        # Read formal charges.
        c = [int(x) for x in next(f).split()]
        # Read bond matrix.
        b = []
        for i in range(n):
            b.append([float(x) for x in next(f).split()])
        # Read distance matrix.
        d = []
        for i in range(n):
            d.append([float(x) for x in next(f).split()])
        a = numpy.array(a, dtype=numpy.float32);
        c = numpy.array(c, dtype=numpy.float32)
        b = numpy.array(b, dtype=numpy.float32)
        d = numpy.array(d, dtype=numpy.float32)
        return (a, c, b, d)


def read_map(name):
    items = []
    with open(name, 'r') as f:
        for line in f:
            line = line.split()
            # Read number of inputs
            ni = int(line[0])
            line = line[1:]
            # Read inputs
            inputs = line[0:ni]
            line = line[ni:]
            # Read number of outputs
            no = int(line[0])
            line = line[1:]
            # Read outputs
            outputs = line[0:no]
            line = line[no:]
            # Read number of tags
            nt = int(line[0])
            line = line[1:]
            # Read tags
            tags = line[0:nt]
            line = line[nt:]
            items.append( [inputs, outputs, tags] )
        return items


def read_nhg(name):
    with open(name, 'rb') as f:
        # Read edge count and node count.
        ne = struct.unpack('i',f.read(4))[0]
        nn = struct.unpack('i',f.read(4))[0]
        # Read nodes (atoms {type,bind-state,x,y,z}).
        n = f.read(nn*5*4)
        nodes = numpy.frombuffer(n, dtype=numpy.float32)
        nodes = numpy.reshape(nodes, (nn,5), order='C')
        # Read edges ({d,n_i,n_j}).
        e = f.read(ne*3*4)
        edges = numpy.frombuffer(e, dtype=numpy.float32)
        edges = numpy.reshape(edges, (ne,3), order='C')
        return nodes, edges


############################################################


def main():
    #
    # Check for a command-line arg.
    #
    if len(sys.argv) != 2:
        print("usage:\n\tchemio.py <voxfile|ligfile|mapfile|nhgfile>")
        return
    #
    # Voxel file?
    #
    if sys.argv[1].endswith('.vox'):
        print("Read Voxels")
        c,t,x,y,z,v = read_vox(sys.argv[1])
        print("  Channels: %d (%s)"%(c,t))
        print("  Size:    ",v.shape)
        print("  Pos:      (%d, %d, %d)"%(x,y,z))
        print("  Fill Vox: %d"%(numpy.count_nonzero(v)))
        return
    #
    # Ligand file?
    #
    if sys.argv[1].endswith('.lig'):
        print("Read Ligand")
        a,c,b,d = read_lig(sys.argv[1])
        print("  Atoms: %d"%(numpy.count_nonzero(a)))
        print("  Bonds: %d"%(numpy.count_nonzero(b)/2))
        return
    #
    # Map (data set) file?
    #
    if sys.argv[1].endswith('.map'):
        print("Read Map")
        items = read_map(sys.argv[1])
        print("  Items:   %d"%(len(items)))
        if len(items):
            print("  Inputs:  %d"%(len(items[0][0])))
            print("  Outputs: %d"%(len(items[0][1])))
            print("  Tags:    %d"%(len(items[0][2])))
            print("  Example: %s"%(str(items[0])))
        return
    #
    # Protein graph file?
    #
    if sys.argv[1].endswith('.nhg'):
        print("Read Protein Neighborhood Graph")
        n,e = read_nhg(sys.argv[1])
        print("  Nodes: %s"%(str(n.shape)))
        for i in range(0,min(len(n),3)):
            print("    Node[%d]:  %s"%(i,str(n[i])))
        print("  Edges: %s"%(str(e.shape)))
        for i in range(0,min(len(e),3)):
            print("    Edge[%d]:  %s"%(i,str(e[i])))
        return


if __name__ == "__main__":
    main()


############################################################
