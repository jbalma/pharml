#!/usr/bin/env python
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

#############################################################################
# The following description is based on Deep Mind's Encoder-Core-Decoder GNN
# The MH-GNN model we explore includes four components:
#
# - An "Encoder" graph net, which independently encodes the Molecular graph's 
#   edge, node, and global attributes.
#
# - An expanding Highway "Core" graph net, which performs N
#   steps. The input to the Core is the concatenation of the Encoder's output
#   and the outputs of the previous Cores (labeled "Hidden({i:i<t})" below, 
#   where "t" is the processing step).
#
# - A "Decoder" graph net, which independently decodes the edge, node, and
#   global attributes (does not compute relations etc.) of the last of the
#   message-passing steps.
# - A "Classification" layer, which is fully-connected (FC) and connects to 
#   the final bind and no-bind class neurons
#
#                 Hidden({i:i<t})  Hidden(t)
#                        |            ^                  |     |
#           *---------*  |  *---------*  |  *---------*  |     |
#           |         |  |  |         |  |  |         |  |     |   |-> NO-BIND
# Input --->| Encoder |  *->| MH-Core |--*->| Decoder |--|  FC | ->|
#           |         |---->|   xN    |     |         |  |     |   |-> BIND
#           *---------*     *---------*     *---------*  |     |
#                                                        |     |
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sonnet as snt
import networkx as nx
import graph_nets as gn
from graph_nets import utils_np
from graph_nets import utils_tf
import gnn_models as models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
import numpy as np

import dataset_utils as du


############################################################


SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)
DEBUG = False
MODE = 'classification'
TARGET_MODE = 'globalstate'
INFERENCE_ONLY = False
INFERENCE_OUT = None
DATA_THREADS = 1
DTYPE=tf.float32
RANK = 0
RANKS = 1
BATCH_SIZE=1
BATCH_SIZE_TEST=1
PRINT_EVERY = 10
############################################################


def plot_history(x_vals, lst_train, lst_test, plot_label, ylabel):
    #Summary and Plots
    plt.clf()
    plt.grid(True)
    plt.plot(x_vals,lst_train)
    plt.plot(x_vals,lst_test)
    plt.title(plot_label)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend(['train', 'test'], loc='upper left')
    plot_name = './'+plot_label+'.svg'
    plt.savefig(plot_name, bbox_inches='tight')


def average_distributed_metrics(sess, acc, solved, loss):
    #Average over workers metrics when using horovod
    import horovod.tensorflow as hvd
    avg_acc = tf.cast(acc,tf.float32)
    avg_sol = tf.cast(solved,tf.float32)
    avg_loss = tf.cast(loss,tf.float32)
    avg_acc_op = hvd.allreduce(avg_acc)
    avg_sol_op = hvd.allreduce(avg_sol)
    avg_loss_op = hvd.allreduce(avg_loss)
    acc = sess.run(avg_acc_op)
    solved = sess.run(avg_sol_op)
    loss = sess.run(avg_loss_op)
    
    #return avg_acc_op.run(), avg_sol_op.run(), avg_loss_op.run()
    return acc, solved, loss


def banner_print(string):
    if DEBUG:
        sep = "----------------------------------------"
        print("%s\n%s\n%s"%(sep,string,sep))


def compute_accuracy_class(target, output):
    if(TARGET_MODE=='globalstate'):
        tds = utils_np.graphs_tuple_to_data_dicts(target)
        solved = [ int(np.argmax(t["globals"]) == np.argmax(o)) for t,o in zip(tds,output[0]) ]
    if(TARGET_MODE=='activesite'):
        tds = utils_np.graphs_tuple_to_data_dicts(target)
        ods = utils_np.graphs_tuple_to_data_dicts(output)
        solved = [ int(np.argmax(t["nodes"]) == np.argmax(o)) for t,o in zip(tds,ods) ]
    return sum(solved), len(solved)


def compute_accuracy_reg(target, output):
    tds = utils_np.graphs_tuple_to_data_dicts(target)
    solved = []
    for td, od in zip(tds, output[0]):
        solved.append(np.abs(td["globals"][0] - od[0]))
    return sum(solved), len(solved)



def write_predictions(items,outputs):
    # Skip if no output file.
    if INFERENCE_OUT is None:
        return
    # Setup the output file once.
    try:
        write_predictions.of = write_predictions.of
    except:
        write_predictions.of = open(INFERENCE_OUT,"w")        
    # Write out all items and predictions in the batch.
    outs = outputs[0]
    for item, out in zip(items,outs):
        nhg_fn, lig_fn, target = item
        nhg_fn = "../nhg/" + os.path.basename(nhg_fn)
        lig_fn = "../lig/" + os.path.basename(lig_fn)
        target = np.argmax(target)
        tag = "bind" if target == 1 else "nobind"
        pred = "pred:%d"%(np.argmax(out))
        write_predictions.of.write("2 %s %s 1 %f 2 %s %s\n"%(nhg_fn,lig_fn,target,tag,pred))
    # Flush.
    write_predictions.of.flush()
    

def item_batch_iter(items, batch_size, shuffle=True):
    # Create data loading threads for the set of items.
    data_threads = du.DataLoader(items,batch_size,nthreads=DATA_THREADS,shuf=shuffle)
    #data_threads = du.DataLoader(items,batch_size,nthreads=DATA_THREADS,shuf=shuffle)
    data_threads.start()
    # Init batch of input / target items.
    it, pd, ld, td = ([], [],[],[])
    # Process all items. 
    elapsed = 0.0
    for item in range(len(items)):
        # If full batch, yield it and reset batch.
        if len(pd) == batch_size:
            yield (it, pd, ld, td)
            it, pd, ld, td = ([], [],[],[])
        # Add new item to current batch.
        start_time = time.time()
        i,p,l,t = data_threads.read_item()
        it.append(i)
        pd.append(p)
        ld.append(l)
        td.append(t)
        elapsed += time.time() - start_time
    # Yield the last (partial) batch.
    yield (it, pd, ld, td)
    # Join with data threads.
    data_threads.join()
    if DEBUG:
        print("\n    Wait:  %.2fs"%elapsed)


def run_batches(sess, batch_generator, input_p_ph, input_l_ph, target_ph, input_p_op, input_l_op, target_op, output_ops, step_op, loss_op):
    #import pdb; pdb.set_trace()
    if tf.get_default_session() != None:
        session = tf.get_default_session()
    else:
        session = sess
    # Init counters / stats.
    start_time = time.time()
    solved, count, loss = (0.0, 0.0, 0.0)
    # Process data in batches.
    if DEBUG:
        sys.stdout.write("    Batch x%s:"%str(PRINT_EVERY))
        sys.stdout.flush()
    for b, batch in enumerate(batch_generator()):
        items, input_dicts_p, input_dicts_l, target_dicts = batch
        # Progress print.
        #if DEBUG and b % 10 == 0 and b>0:
        if ( (b % PRINT_EVERY == 0) and (b>0) ):
            if INFERENCE_ONLY:
                examples_seen=b*BATCH_SIZE_TEST
            else: 
                examples_seen=b*BATCH_SIZE
              
            elapsed_time=time.time()-start_time
            plpps = float(examples_seen)/elapsed_time
            
            if DISTRIBUTED:
                import horovod.tensorflow as hvd
                avg_plpps = tf.cast(plpps,tf.float32)
                avg_plpps_op = hvd.allreduce(avg_plpps)
                plpps = sess.run(avg_plpps_op)
                gplpps = plpps*hvd.size()
                if DEBUG:
                    print("Average Protein-Ligand Pairs Per Second = ", plpps, ", Global = ", gplpps)
                    sys.stdout.flush()
            else: 
                if DEBUG:
                    print("Protein-Ligand Pairs Per Second = ", plpps)
                    sys.stdout.flush()
        # Convert data graph dicts to graphs tuple objects
        input_graphs_p = utils_np.data_dicts_to_graphs_tuple(input_dicts_p)
        input_graphs_l = utils_np.data_dicts_to_graphs_tuple(input_dicts_l)
        target_graphs = utils_np.data_dicts_to_graphs_tuple(target_dicts)


        # Build a feed dict for the data.
        input_p_feed_dict = utils_tf.get_feed_dict(input_p_ph, input_graphs_p)
        input_l_feed_dict = utils_tf.get_feed_dict(input_l_ph, input_graphs_l)
        target_feed_dict = utils_tf.get_feed_dict(target_ph, target_graphs)
        feed_dict = dict()
        feed_dict.update(input_p_feed_dict)
        feed_dict.update(input_l_feed_dict)
        feed_dict.update(target_feed_dict)
        # Run it.
        ops = { "input_p": input_p_op, "input_l": input_l_op, "target": target_op, "loss": loss_op, "outputs": output_ops }
        if step_op != None:
            ops["step"] = step_op
        run_values = session.run(ops, feed_dict=feed_dict)
        # Accumulate stats.
        if MODE == 'classification':
            s, c = compute_accuracy_class(run_values["target"], run_values["outputs"])
        if MODE == 'regression':
            s, c = compute_accuracy_reg(run_values["target"], run_values["outputs"])
        solved, count, loss = (solved+s, count+c, loss+run_values["loss"])
        # If there is an output list, save outputs.
        write_predictions(items, run_values["outputs"])
    elapsed = time.time() - start_time
    # Return stats.
    return elapsed, solved, loss, count


############################################################


def parse_args():
    global DEBUG
    global MODE
    global INFERENCE_ONLY
    global INFERENCE_OUT
    global DATA_THREADS
    global RANK
    global RANKS
    global DTYPE
    #global hvd
    global DISTRIBUTED
    global BATCH_SIZE
    global BATCH_SIZE_TEST
    # Parse command line args.
    parser = argparse.ArgumentParser(prog='ML-Dock-GN using Tensorflow + graph_nets Backend', description='Processing input flags for Training Run.')
    parser.add_argument('--batch_size', type=int, default=4, help='The (local) minibatch size used in training.')
    parser.add_argument('--batch_size_test', type=int, default=8, help='The (local) minibatch size used in testing.')
    parser.add_argument('--map_train', type=str, required=True, help='Path to .map file for training set.')
    parser.add_argument('--map_test', type=str, required=True, help='Path to .map file for test set.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--mlp_layers', type=str, default="4,4", help='Number of layers in each MLP.')
    parser.add_argument('--mlp_latent', type=str, default="32,16", help='Number of neurons in each MLP layer.')
    parser.add_argument('--num_features', type=str, default="64,64", help='Number of output protein features, ligand features.')
    parser.add_argument('--gnn_layers', type=str, default="4,8", help='Number of message passing steps.')
    parser.add_argument('--lr_init', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--hvd', type=bool, default=False, help='Enable the use of Horovod.')
    parser.add_argument('--debug', type=bool, default=True, help='Enable debug tests / prints.')
    parser.add_argument('--use_clr', type=bool, default=False, help='Use Cyclic Learning Rate if true else constant.')
    parser.add_argument('--inference_only', type=bool, default=False, help='Ignore training, only use test set.')
    parser.add_argument('--inference_out', type=str, default=None, help='Write test set predictions to file.')
    parser.add_argument('--data_threads', type=int, default=1, help='Number of data loading threads.')
    parser.add_argument('--mode', type=str, default="regression", help='Training mode: "regression" or "classification".')
    parser.add_argument('--restore', type=str, default=None, help='Path to checkpoint file.')
    parser.add_argument('--plot_history', type=bool, default=False, help='Save training/testing history images')
    parser.add_argument('--use_fp16', type=bool, default=False, help='Use half-precision (tf.float16)')
    args = parser.parse_args()
    DEBUG = args.debug
    MODE = args.mode
    INFERENCE_ONLY = args.inference_only
    INFERENCE_OUT = args.inference_out
    DATA_THREADS = args.data_threads
    DTYPE = tf.float16 if args.use_fp16 else tf.float32
    BATCH_SIZE=args.batch_size
    BATCH_SIZE_TEST=args.batch_size_test
    print(args)
    if args.hvd:
        print("Starting horovod...")
        import horovod.tensorflow as hvd
        #hvd = hvd_temp
        hvd.init()
        RANK = hvd.rank()
        RANKS = hvd.size()
        DISTRIBUTED=True
        print("Initialization of horovod complete...")
        #Index the output filenames for inference output data by rank ID
        if(args.inference_out != None):
            INFERENCE_OUT = str(args.inference_out).split(".")[0] + "_%s.map"%str(RANK)
            print("Rank %s"%str(RANK), " is saving inference output to %s"%str(INFERENCE_OUT))

    if RANK != 0:
        #Only rank 0 should print debug info
        DEBUG = False
        #Reduce logging for all ranks other than 0
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #
    banner_print("PharML.Bind-GNN: Version 1.0.1 - Framework for Open Therapeutics with Graph Neural Networks.")
    banner_print("============================================================================================")
    banner_print("  Developed by")
    banner_print("      Jacob Balma: jb.mt19937@gmail.com")
    banner_print("      Aaron Vose:  avose@aaronvose.net")
    banner_print("      Yuri Petersen: yuripeterson@gmail.com")
    banner_print("This work is supported by collaboration with Cray, Inc, Medical University of South Carolina (MUSC) and Hewlett Packard Enterprise (HPE). ")
    banner_print("============================================================================================")
    if DEBUG:
        print(args)
    # Return parsed args.
    return args


def load_map_items(args):
    # Load data and build needed train / test datasets.
    banner_print("Loading data.")
    train_items = du.load_map_file(args.map_train, RANK, RANKS, TARGET_MODE, debug=DEBUG)
    test_items = du.load_map_file(args.map_test, RANK, RANKS, TARGET_MODE, debug=DEBUG)
    return train_items, test_items


def build_gnn(args,input_shapes):
    # Setup the GNN model.
    banner_print("Building model.")
    tf.reset_default_graph()
    num_features = args.num_features.strip().split(",")
    num_features_p, num_features_l = int(num_features[0]), int(num_features[1])
    gnn_layers = args.gnn_layers.strip().split(",")
    gnn_layers_p, gnn_layers_l = int(gnn_layers[0]), int(gnn_layers[1])
    mlp_latent = args.mlp_latent.strip().split(",")
    mlp_latent_p, mlp_latent_l = int(mlp_latent[0]), int(mlp_latent[1])
    mlp_layers = args.mlp_layers.strip().split(",")
    mlp_layers_p, mlp_layers_l = int(mlp_layers[0]), int(mlp_layers[1])
    gnn_model_p = models.EncodeProcessDecode(edge_output_size=None, node_output_size=None, global_output_size=num_features_p,
                                             mlp_latent=mlp_latent_p, mlp_layers=mlp_layers_p,
                                             num_processing_steps=gnn_layers_p,
                                             name="gnn_model_protein")
    gnn_model_l = models.EncodeProcessDecode(edge_output_size=None, node_output_size=None, global_output_size=num_features_l,
                                             mlp_latent=mlp_latent_l, mlp_layers=mlp_layers_l,
                                             num_processing_steps=gnn_layers_l,
                                             name="gnn_model_ligand")
    ip, il, it = input_shapes
    inputs_p_ph = utils_tf.placeholders_from_data_dicts([ip], force_dynamic_num_graphs=True)
    inputs_l_ph = utils_tf.placeholders_from_data_dicts([il], force_dynamic_num_graphs=True)
    targets_ph  = utils_tf.placeholders_from_data_dicts([it], force_dynamic_num_graphs=True)
    inputs_p_op = utils_tf.make_runnable_in_session(inputs_p_ph)
    inputs_l_op = utils_tf.make_runnable_in_session(inputs_l_ph)
    targets_op  = utils_tf.make_runnable_in_session(targets_ph)
    if DEBUG:
        print("Input protein placeholder = ", inputs_p_ph)
        print("Input ligand placeholder = ", inputs_l_ph)
        print("Target placeholder = ", targets_ph)
    output_p_ops = gnn_model_p(inputs_p_ph)
    output_l_ops = gnn_model_l(inputs_l_ph)
    merged_output = tf.concat([output_p_ops[0].globals,output_l_ops[0].globals],axis=-1)
    initers = { "w": tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                "b": tf.truncated_normal_initializer(stddev=1.0)}
    regs = { "w": tf.contrib.layers.l1_regularizer(scale=0.1),
             "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
    classifier0 = snt.Linear(num_features_p*num_features_l, initializers=initers, regularizers=regs)
    classifier1 = snt.Linear(num_features_p+num_features_l,initializers=initers, regularizers=regs)
    final_output = snt.Linear(2, initializers=initers, regularizers=regs)
    output_ops = [final_output(tf.nn.relu(classifier1(tf.nn.relu(tf.nn.dropout(classifier0(tf.nn.relu(merged_output)),keep_prob=0.5)))))]
    if MODE == 'classification':
        loss_ops = [ tf.losses.softmax_cross_entropy(targets_ph.globals, output_ops[0]) ]
    elif MODE == 'regression':
        loss_ops = [ tf.losses.mean_squared_error(targets_ph.globals, output_ops[0]) ]
    loss_op = sum(loss_ops)

    # Return ops and placeholders
    return inputs_p_ph, inputs_l_ph, targets_ph, inputs_p_op, inputs_l_op, targets_op, output_ops, loss_op


def build_optimizer(args,loss_op,num_train_items):
    # Optimizer.
    banner_print("Building optimizer.")
    global_step = tf.train.get_or_create_global_step()
    lr_init = float(RANKS*args.lr_init)
    if args.use_clr:
        import clr
        if RANK == 0:
            print("Using Cyclic LR with initial LR: ", lr_init)
        step_sz = num_train_items/args.batch_size
        max_steps = args.epochs*(num_train_items/args.batch_size)
        lr_decay = tf.train.exponential_decay(lr_init, global_step, max_steps*10, 0.9, staircase=False)
        learning_rate = clr.cyclic_learning_rate(global_step=global_step, learning_rate=lr_decay, max_lr=100*lr_decay,
                                                 step_size=2*step_sz, mode='triangular', gamma=.999)

    else:
        learning_rate = lr_init
        if RANK == 0:
            print("Using constant LR: ", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate) 
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    if args.hvd:
        import horovod.tensorflow as hvd
        compression = hvd.Compression.fp16 if args.use_fp16 else hvd.Compression.none
        
        optimizer = hvd.DistributedOptimizer(optimizer, use_locking=False, compression=compression, op=hvd.Average)
        optimizer._learning_rate = tf.cast(learning_rate,tf.float32)
    else:
        optimizer._learning_rate = tf.cast(learning_rate,tf.float32)
    step_op = optimizer.minimize(loss_op,global_step)
    # Return the optimizer and the step_op
    return optimizer, step_op


def run_gnn(args,model_ops,test_items,train_items=None,optimizer=None):
    # Split ops.
    inputs_p_ph, inputs_l_ph, targets_ph, inputs_p_op, inputs_l_op, targets_op, output_ops, loss_op, step_op = model_ops
    # Create new TF session.
    banner_print("Create TF config / session.")
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    if args.hvd:
        import horovod.tensorflow as hvd
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        #Some versions of tensorflow(version<1.15) have issues with allocating all device memory
        #In this case, uncomment the following line
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        checkpoint_dir = './checkpoints' if RANK == 0 else './checkpoints_test'
    else:
        config.gpu_options.visible_device_list = str(0)
        checkpoint_dir = './checkpoints'
    try:
        os.mkdir(checkpoint_dir)
    except OSError:  
        print ("Creation of directory %s failed!"%checkpoint_dir)
    else:  
        print ("Successfully created directory %s."%checkpoint_dir)
    sess = tf.Session(config=config)
    # Initialize Model.
    if(RANK==0):
        print("All workers are initializing global variables...")
    #All ranks should initialize their variables before 
    #loading checkpoints or getting broadcast variables 
    #from rank 0
    sess.run(tf.global_variables_initializer())
    if(RANK==0):
        print("Done global variables init.")

    saver = tf.train.Saver()
    model_path = checkpoint_dir + '/model%s.ckpt'%RANK
    restore_path = model_path
    
    if RANK==0:
        # Test Save / Restore model with Rank 0.
        save_path = saver.save(sess, model_path)
        print("Coordinator Test checkpoint saved to: %s"%save_path)
    else:
        save_path = saver.save(sess, model_path)
        print("Worker test checkpoint saved to: %s"%save_path)

    
    if args.restore != None:
        restore_path = args.restore
        print("Restoring model from: %s"%restore_path)
        saver.restore(sess,restore_path)
        print("Model restored sucessfully.")
        print("To resume training use --restore %s"%str(os.getcwd()+"/"+restore_path))
    else:
        restore_path=model_path
        saver.restore(sess, restore_path)
        print("Worker fresh checkpoint restore test success.")
        print("To resume training use --restore %s"%str(os.getcwd()+"/"+restore_path))
        print("Training new model.")

    # Print total model parameters.
    if 0:
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total trainable params: ", total_parameters)

    if args.hvd:
        print("Broadcasting...")
        import horovod.tensorflow as hvd
        bcast_op = hvd.broadcast_global_variables(0)
        sess.run(bcast_op)
        time.sleep(10)
        print("Done broadcast")


    # Training / inference loop.
    banner_print("Start training / testing loop.")
    acc_best = 0.0
    epoch_best = 0
    log_epochs, solveds_tr, solveds_ge, losses_tr, losses_ge, lr_hist = ([],[],[],[],[],[])
    for epoch in range(args.epochs):
        if RANK == 0:
            print("Epoch %d:"%(epoch))
            log_epochs.append(epoch)
        # Run training step.
        if not INFERENCE_ONLY:
            if RANK == 0:
                print("  Training.")
            elapsed, solved, loss, count = run_batches(sess, lambda: item_batch_iter(train_items,args.batch_size),
                                                       inputs_p_ph, inputs_l_ph, targets_ph,
                                                       inputs_p_op, inputs_l_op, targets_op,
                                                       output_ops, step_op, loss_op)
            acc = solved / count
            loss = loss / count
            lr = sess.run(optimizer._learning_rate)
            if args.hvd:
                acc, solved, loss = average_distributed_metrics(sess, acc, solved, loss)
                count = hvd.size()*count
                solved = hvd.size()*solved
            if RANK == 0:
                print("    Time:  %.1fs"%(elapsed))
                print("    LrnR:  %.6f"%lr)
                print("    Loss:  %f"%(loss))
                print("    Acc.:  %f  (%.1f/%.1f)"%(acc,solved,count))
                solveds_tr.append(acc)
                losses_tr.append(loss)
                lr_hist.append(lr)
        # Run a test step.
        if RANK == 0:
            print("  Testing.")
        elapsed, solved, loss, count = run_batches(sess, lambda: item_batch_iter(test_items,args.batch_size_test, shuffle=False),
                                                   inputs_p_ph, inputs_l_ph, targets_ph,
                                                   inputs_p_op, inputs_l_op, targets_op,
                                                   output_ops, None, loss_op)
        acc = solved / count
        loss = loss / count
        if args.hvd:
            acc, solved, loss = average_distributed_metrics(sess, acc, solved, loss)
            count = hvd.size()*count
            solved = hvd.size()*solved
        if RANK == 0:
            print("    Time:  %.1fs"%(elapsed))
            print("    Loss:  %f"%(loss))
            print("    Acc.:  %f  (%.1f/%.1f)"%(acc,solved,count))
            solveds_ge.append(acc)
            losses_ge.append(loss)
            if(args.plot_history):
                plot_history(log_epochs, solveds_tr, solveds_ge, 'PharML-Accuracy', 'accuracy')
                plot_history(log_epochs, losses_tr, losses_ge, 'PharML-Loss','loss')
                plot_history(log_epochs, lr_hist, lr_hist, 'PharML-LR','learning rate')

        # Checkpoint if needed.
        if acc > acc_best and not INFERENCE_ONLY:
            acc_best = acc
            epoch_best = epoch
            if RANK == 0:
                print("  New Best Test Acc: ", acc_best)
                print("   -> Occurred at epoch ", epoch_best)
                sys.stdout.flush()
                save_path = saver.save(sess, model_path)
                print("   -> Saved checkpoint to %s"%(save_path))

        if INFERENCE_ONLY:
            # Exit loop after first inference if in inference only-mode.
            print("Inference only mode, done with single pass so exiting...")
            hvd.shutdown()
            break;

        # If test accuracy has not improved for more than 
        # 15 epochs, call it converged and exit - this is what 
        # was used in paper, but since we've found lowering 
        # this to 5 epochs is sufficient in some cases
        if( (epoch-epoch_best) >= 15 and not INFERENCE_ONLY):
            print("Model Converged! Exiting Nicely...")
            #sys.exit(0)
            hvd.shutdown()
            break;
            
    # Success!
    banner_print("Success!")


############################################################


def main():
    # Parse command-line args.
    args = parse_args()
    # Load map files.
    train, test = load_map_items(args)
    # Load one item set for their shape.
    protein, ligand, target = du.load_item(train[0])
    # Build the GNN model.
    ops = build_gnn(args,(protein,ligand,target))
    # Setup the training optimizer
    if INFERENCE_ONLY:
        #optimizer = None
        optimizer, step_op = build_optimizer(args,ops[-1],len(test))
        ops += (step_op,)
    else:
        optimizer, step_op = build_optimizer(args,ops[-1],len(train))
        ops += (step_op,)
    # Run the training / inference loop.
    run_gnn(args, ops, test, train_items=train, optimizer=optimizer)

    
if __name__== "__main__":
    main()


############################################################
