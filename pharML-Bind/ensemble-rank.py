#!/usr/bin/env python
############################################################
"""
Copyright 2020 Hewlett Packard Enterprise Development LP and
MUSC foundation for Research Development

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import argparse
import numpy as np

from chemio import read_map 


############################################################


def ensemble_maps(maps):
    # Do a sanity-check on item0.
    for m in maps:
        first_item = m[0] #[inputs, outputs, tags]
        print("first item = ", first_item)
        if len(first_item[0]) != 2:
            print("Expected two inputs (NHG and LIG)!")
            sys.exit(1)
        if len(first_item[1]) != 1:
            print("Expected one output (bind/nobind)!")
            sys.exit(1)
        if len(first_item[2]) != 2:
            print("Expected 2 tags (actual, prediction)")
            sys.exit(1)
        #if len(first_item[3]) !=2: 
        #    print("Expected 2 probabilities for no-bind and bind!")
        #    sys.exit(1)
    # Bin all items for each map by (p,l) pair.
    dmaps = [ {} for m in maps ]
    for mndx,m in enumerate(maps):
        for item in m:
            inputs, outputs, tags = item
            dmaps[mndx][tuple(inputs)] = [outputs,tags]
        print("  Map%d: %d items"%(mndx,len(dmaps[mndx])))
    # Make sure maps have the same items.
    for i,idmap in enumerate(dmaps):
        for ikey in idmap:
            for j,jdmap in enumerate(dmaps):
                if ikey not in jdmap:
                    print("Item from map %d not in map %d!"%(i,j))
                    sys.exit(1)
    # Each item has "[outputs,tags]", augment with ensemble info.
    ensemble = {}
    for key in dmaps[0]:
        actual = 1.0 if float(dmaps[0][key][0][0]) == 1.0 else 0.0
        binds = 0
        out = [0.0,0.0]
        for dmap in dmaps:
            outputs, tags = dmap[key]
            raw_out = []
            for t in tags:
                raw_out.append(t.strip("nb:").strip("b:").strip(","))
            fp_out = [0.0,0.0]
            fp_out[0] = float(raw_out[0])
            fp_out[1] = float(raw_out[1])
            #print("raw outs=", raw_out, ", fp_out=",fp_out)
            prediction = float(np.argmax(fp_out))
            #prediction = 1.0 if "pred:1" in tags else 0.0
            if prediction == 1.0:
                binds += 1
                out[0] += fp_out[0]
                out[1] += fp_out[1]
        ensemble[key] = (key, actual, binds, out)
    # Get some totals / global stats.
    total_binds = 0
    total_nobinds = 0
    for key in ensemble:
        k, actual, binds, out = ensemble[key]
        if actual == 1.0:
            total_binds += 1
        else:
            total_nobinds += 1
    print("  actual: %.2f%% binds (%d/%d)"%(100.0*(total_binds/(total_binds+total_nobinds)),total_binds,total_nobinds))
    # Build a dict based on bind prediction counts.
    counts = { i:[] for i in reversed(range(len(dmaps)+1)) }
    for item in sorted(ensemble.values(), key=lambda e: e[2]):
        key, actual, binds, out = item
        #print("working on building dict for key=",key)
        #print(" -> has raw out = ", out)
        counts[binds].append(item)

    top_counts = { i:[] for i in reversed(range(len(dmaps)+1)) }
    for item in sorted(counts[4], key=lambda e: e[3][1]):
        #print("item = ", item)
        key, actual, binds, out = item
        #print("working on sorting dict for key=",key)
        #print(" -> has raw out = ", out)
        #print(" -> and binds = ", binds)
        top_counts[binds].append(item)

    f= open("top-preds.txt","w+")
    for item in reversed(sorted(counts[5], key=lambda e: e[3][1])):
        key, actual, binds, out = item
        cid = key[1].strip("lig").strip("\.").strip("\/") #key[1].split(".lig")[0].split(".*.lig")[1].strip(" ")
        cid = cid.split('lig')[-1]
        #print("top cid=", cid)
        print("%s %s %s %s"%(cid,binds,out[0],out[1]))
        f.write("%s %s %s %s \n"%(cid,binds,out[0],out[1]))
    f.close()

    # Print ensemble info for each prediction count.
    cum_cbinds = 0
    cum_ncount = 0
    for count in counts:
        ncount = len(counts[count])
        ncpct = 100.0 * ncount / len(maps[0])
        cbinds = 0
        for item in counts[count]:
            key, actual, binds, out = item
            if actual == 1.0:
                cbinds += 1
        bpct = 100.0 * cbinds / ncount
        cum_cbinds += cbinds
        cum_ncount += ncount
        cbpct = 100.0 * cum_cbinds / cum_ncount
        if total_binds == 0:
            IEF = -1.0
            EF = -1.0
        else:
            IEF = 1.0 / (float(total_binds) / (total_binds+total_nobinds))
            EF = IEF * (float(cum_cbinds) / cum_ncount) 
        print("  count%d: %d  (%.1f%%)  actual_bind%%: %.1f (%.1f cum)  EF(cum): %.2f (ideal %.2f)"%(count,ncount,ncpct,bpct,cbpct,EF,IEF))
    print("  count*: %d  (100.0%%)"%(len(maps[0])))
    return

############################################################


def parse_args():
    # Parse command line args.
    parser = argparse.ArgumentParser(prog='ensemble.py: Combine inference outputs into an ensemble.', description='Combine inference outputs into an ensemble.')
    parser.add_argument('--maps', type=str, required=True, help='Colon-seperated list of .map file paths.')
    parser.add_argument('--out', type=str, required=True, help='File to save the lig IDs (CHEMBL).')
    args = parser.parse_args()
    print(args)
    # Return parsed args.
    return args


def main():
    # Parse command-line args.
    args = parse_args()
    map_fns = args.maps.split(':')
    # Verbose print.
    print("Creating ensemble from %d map files:"%(len(map_fns)))
    for map_fn in map_fns:
        print("  %s"%(map_fn))
    # Read the map files.
    maps = [ read_map(map_fn) for map_fn in map_fns ]
    # Write out a map file representing the ensemble.
    ensemble_maps(maps)
    print("Success!")

    
if __name__== "__main__":
    main()


############################################################
