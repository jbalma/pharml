#!/usr/bin/env python
############################################################


import argparse
import sys
import time
import numpy
from PIL import Image
import tensorflow as tf

from mldock_gnn import build_gnn
from mldock_gnn import run_batches
from dataset_utils import load_protein_file
from ligand import Ligand


############################################################


WORST_ERROR = 1.0e20
VERSION = "0.0.2"


############################################################


def write_gif(images,path):
    # Create animated image.
    images = [ Image.open(fn) for fn in images ]
    images[0].save(path,
                   save_all=True,
                   append_images=images[1:],
                   duration=300,
                   loop=0)


############################################################


def batch_population(pop,protein,batch_size=4):
    target = numpy.array([0.0,1.0],dtype=numpy.float32)
    target = {'n_node':0, 'n_edge':0, 'nodes':None,
              'edges':None, 'senders':None, 'receivers':None,
              'globals':target}
    # Init batch of input / target items.
    pd, ld, td = ([],[],[])
    # Process all pop members. 
    batch = 0
    for item in range(len(pop)):
        # If full batch, yield it and reset batch.
        if len(pd) == batch_size:
            if batch % 10 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            batch += 1
            yield (pd, ld, td)
            pd, ld, td = ([],[],[])
        # Add new item to current batch.
        p,l,t = (protein, pop[item], target)
        pd.append(p)
        ld.append(l)
        td.append(t)
    # Yield the last (partial) batch.
    yield (pd, ld, td)
    print("!")


def eval_inds(pop,gnn,protein):
    # Convert inds to packed list of valid data dicts for the gnn.
    popd = [ ind.get_dict() for ind in pop ]
    full_pr = [ -1.0 if ind is not None else WORST_ERROR for ind in popd ]
    popd = [ ind for ind in popd if ind is not None ]
    print("    NumEval:   %d"%(len(popd)))
    # Run the gnn.
    presults = []
    sess, ops = gnn
    in_p_ph, in_l_ph, targets_ph, in_p_op, in_l_op, targets_op, out_ops, loss_op = ops
    elapsed, solved, loss, count = run_batches(sess, lambda: batch_population(popd,protein),
                                               in_p_ph, in_l_ph, targets_ph,
                                               in_p_op, in_l_op, targets_op,
                                               out_ops, None, loss_op, output=presults)
    # Expand the packed valid list into the full list.
    ldndx = 0
    for result in presults:
        for dndx in range(ldndx,len(full_pr)):
            ldndx = dndx
            if full_pr[dndx] == -1.0:
                full_pr[dndx] = result
                break
    # Return the population's fitnesses.
    return full_pr


def ga_search(args,gnn):
    # Create initial population.
    protein = load_protein_file(args.protein)
    population = [ Ligand(smiles) for smiles in args.initpop.split(',') ]
    population += [ population[0].clone() for i in range(args.popsz-len(population)) ]
    for ind in population[1:]:
        ind.mutate(args.mu)
    # Generation loop.
    print("Start genetic algorithm search (%d):"%args.gens)
    best = population[0]
    best_fit = WORST_ERROR
    images = []
    for gen in range(args.gens):
        # Stdout print.
        print("  Generation %d:"%gen)
        print("    PopSize:   %d"%len(population))
        # Compute fitness.
        tstart = time.time()
        fits = eval_inds(population,gnn,protein)
        tfend = time.time()
        viable = sum(fit != WORST_ERROR for fit in fits)
        # Viability selection.
        if viable is 0:
            print("    Population went extinct!")
            break
        print("    nViable:   %d"%viable)
        indexes = list(range(len(fits)))
        indexes.sort(key=fits.__getitem__)
        fits = list(map(fits.__getitem__, indexes))
        population = list(map(population.__getitem__, indexes))
        population = population[0:int(args.popsz*args.sigma)]
        elite, efits = ([],[])
        for ndx,ind in enumerate(population):
            if ind.smiles not in elite:
                elite.append(ind.smiles)
                efits.append(fits[ndx])
                if len(elite) == 3:
                    break
        population = [ ind for i,ind in enumerate(population) if fits[i] != WORST_ERROR ]
        fits = fits[0:len(population)]
        # Some initial stats.
        if fits[0] < best_fit:
            best_fit = fits[0]
            best = population[0]
            if args.img:
                images.append("%s/best_%d.png"%(args.out,gen))
                best.write_image(images[-1])
                write_gif(images,"%s/best.gif"%(args.out))
        print("    Fitness:   %f"%(sum(fits)/len(fits)))
        print("    BestFit:   %f"%(best_fit))
        print("    Best:      %s"%(best.smiles))
        print("    Elite:")
        for ndx,ind in enumerate(elite):
            print("      (%.4f) %s"%(efits[ndx],ind))
        # Reproduction.
        opop = len(population)
        population.append( population[0].clone() )
        while len(population) < args.popsz:
            # Crossover.
            indi = population[numpy.random.randint(0,opop)]
            indj = population[numpy.random.randint(0,opop)]
            inds = indi.mate(indj)
            # Mutation.
            for ind in inds:
                ind.mutate(args.mu)
                population.append( ind )
        # End of generation timing and print.
        tend = time.time()
        print("    Time:      %.2f s (fit %.3f, ga %.3f)"%(tend-tstart,tfend-tstart,tend-tfend))
    # Return best found.
    return best


############################################################


def init_gnn(args):
    # Setup gnn fitness oracle.
    protein = load_protein_file(args.protein)
    ligand = Ligand("CCC").get_dict()
    target = numpy.array([0.0,1.0],dtype=numpy.float32)
    target = {'n_node':0, 'n_edge':0, 'nodes':None,
              'edges':None, 'senders':None, 'receivers':None,
              'globals':target}
    ops = build_gnn(args,(protein,ligand,target))
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,args.model)
    return (sess, ops)


############################################################


def parse_args():
    # Parse and return command line args.
    parser = argparse.ArgumentParser(description='mldock-evo'+' v'+VERSION)
    parser.add_argument('--model',     required=True,  type=str,   help='GNN model checkpoint.')
    parser.add_argument('--protein',   required=True,  type=str,   help='Target NHG file.')
    parser.add_argument('--initpop',   default="CC",   type=str,   help='Starting SMILES strings.')
    parser.add_argument('--sigma',     default=0.05,   type=float, help='Selection strength.')
    parser.add_argument('--mu',        default=0.1,    type=float, help='Mutation rate.')
    parser.add_argument('--gens',      default=1000,   type=int,   help='Generations.')
    parser.add_argument('--popsz',     default=500,    type=int,   help='Population size.')
    parser.add_argument('--out',       default="data", type=str,   help='Output directory.')
    parser.add_argument('--img',       default=True,   type=bool,  help='Write images.')
    parser.add_argument('--seed',      default=7,      type=int,   help='Random seed.')
    parser.add_argument('--mlp_layers',default="2,2",  type=str,   help='MLP hidden layers.')
    parser.add_argument('--mlp_latent',default="32,32",type=str,   help='MLP hidden neurons.')
    parser.add_argument('--features',  default="16,16",type=str,   help='Protein/ligand features.')
    parser.add_argument('--gnn_layers',default="8,8",  type=str,   help='Message passing steps.')
    args = parser.parse_args()
    args.num_features = args.features
    return args


def main():
    # Parse command-line args.
    args = parse_args()
    # Make the libraries used by the Ligand class shut up.
    Ligand(None).quiet()
    # Set random seed.
    numpy.random.seed(args.seed)
    # Perform the GA search.
    gnn = init_gnn(args)
    # Perform the GA search.
    ga_best = ga_search(args,gnn)
    
    
if __name__== "__main__":
    main()


############################################################
