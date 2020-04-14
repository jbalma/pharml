#!/usr/bin/env python
############################################################

# "Improved Scoring of Ligand-Protein Interactions Using
# OWFEG Free Energy Grids", David A. Pearlman and Paul S.
# Charifson, J. Med. Chem. 2001, 44, 502-511.

import argparse
import os


############################################################


if __name__ == "__main__":
    # Parse command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc',  type=float, required=True,   help='Accuracy.')
    parser.add_argument('--size', type=float, required=True,   help='Test set size.')
    args = parser.parse_args()
    # Write data file.
    with open("test_ef.txt","w") as of:
        of.write("#ideal_frac act_frac enrichment_factor\n")
        steps = 10
        stepsz = 1.0 / float(steps)
        for step in range(1,steps+1):
            step = float(step)
            frac = step*stepsz
            size = args.size
            while size-int(frac*size) > int(0.5*args.size) or int(frac*size) > int(0.5*args.size):
                size = size - 1
                if size == 0:
                    print("Size == 0!")
                    exit()
            binds = float(int(frac*size))
            nobinds = float(int(size - binds))
            num = binds*args.acc
            den = binds*args.acc + nobinds*(1.0-args.acc)
            ef = (num/den) / (binds/(binds+nobinds))
            print(frac,(binds/(binds+nobinds)),binds,nobinds,ef)
            of.write("%f %f %f\n"%(frac,(binds/(binds+nobinds)),ef))
    # Write gnuplot file.
    with open("test_ef.gnuplot","w") as of:
        of.write("set term pngcairo color\n")
        of.write("set out 'test_ef.png'\n")
        of.write("set title 'MLPharm.Bind Enrichment Factor at %.4f Accuracy'\n"%(args.acc))
        of.write("set xlabel 'Test Set Bind Fraction'\n")
        of.write("set ylabel 'Enrichment Factor'\n")
        of.write("plot 1.0 with lines noti, \\\n")
        of.write("     'test_ef.txt' using 2:3 with lines title 'Enrichment Factor'\n")
    # Run gnuplot and view file.
    os.system("gnuplot test_ef.gnuplot")
    os.system("xview test_ef.png")
    # Done!
    print("Success!")
    
    
############################################################
