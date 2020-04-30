#!/usr/bin/python
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

import numpy as np
import sys
import time
import os
import threading
from random import shuffle
from graph_nets import graphs 
from chemio import read_nhg, read_lig, read_map 


############################################################


def print_graphs_tuple(graphs_tuple):
    print(graphs_tuple.map(lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS))
    if graphs_tuple.globals is not None:
        print("globals:\n{}".format(graphs_tuple.globals))
    if graphs_tuple.nodes is not None:
        print("nodes:\n{}".format(graphs_tuple.nodes))
    if graphs_tuple.edges is not None:
        print("edges:\n{}".format(graphs_tuple.edges))
    if graphs_tuple.senders is not None:
        print("senders:\n{}".format(graphs_tuple.senders))
    if graphs_tuple.receivers is not None:
        print("receivers:\n{}".format(graphs_tuple.receivers))
    if graphs_tuple.n_node is not None:
        print("n_node:\n{}".format(graphs_tuple.n_node))
    if graphs_tuple.n_edge is not None:
        print("n_edge:\n{}".format(graphs_tuple.n_edge))


############################################################


def load_map_file(map_path, rank=0, size=1, target_mode='globalstate', debug=False):
    # Start timer for performance data
    if rank == 0:
        print("Loading map file as distributed dataset for %d ranks:"%size)
        start_main = time.time()
    else:
        debug=False 
    # Read the map file.
    if debug:
        print("  Map file:          %s"%(map_path))
    item_map = read_map(map_path)
    chunk_size = int(len(item_map)/size)
    item_map = item_map[:chunk_size*size]
    chunk_start = rank * chunk_size
    chunk_end =  (rank + 1) * chunk_size
    if debug:
        print("  Map items:         %d"%(len(item_map)))
        print("  Worker chunk size: %d"%(chunk_size))
    # Process the training/inference items in the map.
    items = []
    for i,item in enumerate(item_map[chunk_start:chunk_end]):
        # Split apart the item.
        prots  = 'nhg/' + os.path.basename(item[0][0])
        ligand = 'lig/' + os.path.basename(item[0][1]) 
        # Set paths.
        base_dir = os.path.dirname(os.path.abspath(map_path)) + '/../'
        base_dir = os.path.abspath(base_dir) + '/'
        pfile    = base_dir + prots
        lfile    = base_dir + ligand 

        # Depending on target_mode, actual target 
        # values are set to either:
        # 1) target_mode==globalstate: global-level scalar for protein-ligand bind state (bind/nobind for each PLP)
        # 2) target_mode==activesite: node-level graph where each node contains protein's atomic bind-state for given ligand 
        if(target_mode=='globalstate'):
            # Target (bind/nobind) of ligand/protein pair.
            global_state = float(item[1][0])
            target = [0.0,1.0] if global_state == 1.0 else [1.0,0.0]
            target = np.array(target,dtype=np.float32)
        if(target_mode=='activesite'):
            global_state = float(item[1][0])
            target = get_activesite_target(pfile,global_state)
        # Add the item to the list to return.
        items.append( [pfile,lfile,target] )
        # End loop over items in a map.
    # Debug prints.
    if debug:
        finish_main = time.time()
        st0 = sum( [item[2][0] for item in items] )
        print("  Number of items:   %d"%(len(items)))
        print("  Sum(target[0]):    %f"%(st0))
        print("  Avg(target[0]):    %f"%(st0/len(items)))
        print("  Total load time:   %.1f"%(finish_main-start_main))
        print("Done processing map file.")
    # Return list of items
    return items


def load_ligand_file(lfile):
    # Read the raw data.
    a,c,b,d = read_lig(lfile)
    ################################
    # Ligand raw data format:
    #   a = 1D array of atomic charges for each atom
    #   b = 2D matrix of bond types between atoms
    #   c = 1D array of formal charges for each atom
    #   d = 2D matrix of distances between atoms
    # Intended ligand graph format:
    #   nodes (atoms {atom-type, formal-charge}).
    #   edges ({bond, n_i, n_j}).
    ################################
    # Ligand vertices.
    lv  = np.array([ [v[0],v[1]] for v in zip(a,c) ], dtype=np.float32)
    # Ligand edges.
    eal, esl, erl = ([], [], [])
    for j in range(0,b.shape[0]):
        for k in range(0,b.shape[1]):
            if b[j][k] != 0:
                eal.append( [b[j][k]] )
                esl.append(j)
                erl.append(k)
    eal = np.array(eal,dtype=np.float32)
    esl = np.array(esl,dtype=np.float32)
    erl = np.array(erl,dtype=np.float32)
    # Fill in the input data dict.
    gl = np.array([0.0],dtype=np.float32)
    l_dict = {'n_node':lv.shape[0], 'n_edge':eal.shape[0], 'nodes':lv,
              'edges':eal, 'senders':esl, 'receivers':erl, 'globals':gl}
    # Return the data.
    return l_dict


def load_protein_file(pfile):
    # Read the raw data.
    pv_raw,pe_raw = read_nhg(pfile)
    ################################
    # Protein graph format:
    #   nodes (atoms {t,b,x,y,z}).
    #   edges ({distance, n_i, n_j}).
    #   nodes = (protein atoms {atom type,active-site,x,y,z})
    #   edges = (protein atom-neighbors {distance, atom index, neighbor index})
    # Intended protein graph format:
    #   nodes (atoms {atom-type}).
    #   edges ({distance, n_i, n_j}).
    ################################
    # Protein vertices [atom type].
    pv = np.array([ [v[0]] for v in pv_raw ], dtype=np.float32)
    # Protein edges [distance, atom_a, atom_b].
    eap = np.zeros((pe_raw.shape[0],1),dtype=np.float32)
    eap[:,0] = np.reciprocal(pe_raw[:,0])
    esp = pe_raw[:,1]
    erp = pe_raw[:,2]
    # Fill in the input data dict.
    gp = np.array([0.0],dtype=np.float32)
    p_dict = {'n_node':pv.shape[0], 'n_edge':eap.shape[0], 'nodes':pv,
              'edges':eap, 'senders':esp, 'receivers':erp, 'globals':gp}
    # Return the data.
    return p_dict



def get_activesite_target(pfile,bind_state):
    # This function returns a dictionary representation 
    # of a graph which labels each node (atom) of the input protein
    #  ->  as 0.0 everywhere if the global bind state is no-bind
    #  ->  as 0.0 or 1.0 depending on whether it is located within a threshold distance from any ligand atom
    # The global scalar value for the graph is also set to 1.0 or 0.0 depending on the global bind state
    # Read the raw data.
    pv_raw,pe_raw = read_nhg(pfile)
    ################################
    # Protein graph format:
    #   nodes (atoms {t,b,x,y,z}).
    #   edges ({distance, n_i, n_j}).
    #   nodes = (protein atoms {atom-type,active-site,x,y,z})
    #   edges = (protein atom-neighbors {distance, atom index, neighbor index})
    # Intended protein graph format:
    #   nodes (atoms {atom-type,active-site state}).
    #   edges ({distance, n_i, n_j}).
    ################################
    # Protein vertices [atom type,active-site].
    if(bind_state==1):
        pv = np.array([ [v[0],v[1]] for v in pv_raw ], dtype=np.float32)
    else:
        pv = np.array([ [v[0],0.0] for v in pv_raw ], dtype=np.float32)
    # Protein edges are same as input [distance, atom_a, atom_b].
    eap = np.zeros((pe_raw.shape[0],1),dtype=np.float32)
    eap[:,0] = np.reciprocal(pe_raw[:,0])
    esp = pe_raw[:,1]
    erp = pe_raw[:,2]
    # Protein globals are same 
    if(bind_state==1):
        gp = np.array([1.0],dtype=np.float32)
    else:
        gp = np.array([0.0],dtype=np.float32)
    # Fill in the target data dict.
    p_dict = {'n_node':pv.shape[0], 'n_edge':eap.shape[0], 'nodes':pv,
              'edges':eap, 'senders':esp, 'receivers':erp, 'globals':gp}
    # Return the data.
    return p_dict

############################################################


def load_item(item):
    # Parse map item; set target.
    pfile, lfile, target = item
    t_dict = {'n_node':0, 'n_edge':0, 'nodes':None,
              'edges':None, 'senders':None, 'receivers':None,
              'globals':target}
    # Read the ligand and protein data
    l_dict = load_ligand_file(lfile)
    p_dict = load_protein_file(pfile)
    # Return the loaded item.
    return p_dict, l_dict, t_dict


############################################################


class DataLoader(threading.Thread):
    def __init__(self,items_list,buffer_size,nthreads=1,parent=None,shuf=True):
        threading.Thread.__init__(self)
        # Save settings.
        self.map_items = items_list
        self.buffsz = buffer_size
        self.helpers = []
        if parent is not None:
            # Child uses state from parent.
            self.item_buffer = parent.item_buffer
            self.lock = parent.lock
            self.non_empty = parent.non_empty
            self.non_full = parent.non_full
        else:
            # Master shuffles at the start of the epoch.
            self.map_items = list(self.map_items)
            if shuf:
                shuffle(self.map_items)
            # Buffer space.
            self.item_buffer = []
            # Locks and events / signals.
            self.lock = threading.Lock()
            self.non_empty = threading.Event()
            self.non_full = threading.Event()
            self.non_full.set()
            # Spawn child threads.
            for tndx in range(nthreads-1):
                helper = DataLoader(self.map_items,self.buffsz,nthreads=0,parent=self)
                self.helpers.append(helper)
        return

    def read_item(self):
        # Lock for access to read data item.
        self.lock.acquire()
        # Wait until there is data.
        while not self.non_empty.is_set():
            self.lock.release()
            self.non_empty.wait()
            self.lock.acquire()
        # Slice off an item.
        item = self.item_buffer.pop(0)
        # If buffer was full, mark as not full.
        if not self.non_full.is_set():
            self.non_full.set()
        # If the buffer is empty, mark that.
        if len(self.item_buffer) == 0:
            self.non_empty.clear()
        # Unlock.
        self.lock.release()
        # Return the read item.
        return item

    def run(self):
        # Start any needed helper threads.
        for helper in self.helpers:
            helper.start()
        while True:
            # Wait until the buffer is not full.
            self.non_full.wait()
            # Lock so we can get a new map item.
            self.lock.acquire()
            # Make sure there is at least one item.
            if len(self.map_items) == 0:
                # Done.
                self.lock.release()
                # Parent waits on children first.
                for helper in self.helpers:
                    helper.join()
                # Thread exit.
                return
            # Get a map item off the list.
            item = self.map_items.pop(0)
            # Release lock before I/O.
            self.lock.release()
            # Read from disk while not holding a lock.
            p_data, l_data, t_data = load_item(item)
            # Lock so we can fill loaded item into buffer.
            self.lock.acquire()
            # Make sure there is still room in the buffer.            
            while not self.non_full.is_set():
                self.lock.release()
                self.non_full.wait()
                self.lock.acquire()
            # Write the item into the buffer.
            self.item_buffer.append( (item, p_data, l_data, t_data) )
            # If the buffer was empty, mark as not.
            if not self.non_empty.is_set():
                self.non_empty.set()
            # If the buffer is full, mark that.
            if len(self.item_buffer) == self.buffsz:
                self.non_full.clear()
            # Unlock.
            self.lock.release()


############################################################
