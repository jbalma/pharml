#!/usr/bin/env python
############################################################
#
# The model we explore includes three components:
#
# - An "Encoder" graph net, which independently encodes the edge, node, and
#   global attributes (does not compute relations etc.).
#
# - A "Core" graph net, which performs N rounds of processing (message-passing)
#   steps. The input to the Core is the concatenation of the Encoder's output
#   and the outputs of the previous Cores (labeled "Hidden({i:i<t})" below, 
#   where "t" is the processing step).
#
# - A "Decoder" graph net, which independently decodes the edge, node, and
#   global attributes (does not compute relations etc.) of the last of the
#   message-passing steps.
#
#                 Hidden({i:i<t})  Hidden(t)
#                        |            ^
#           *---------*  |  *------*  |  *---------*
#           |         |  |  |      |  |  |         |
# Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output
#           |         |---->|      |     |         |
#           *---------*     *------*     *---------*
#
############################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Concatenate
from keras.models import Model

import sonnet as snt
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
INFERENCE_ONLY = False
INFERENCE_OUT = None
QUANTIZE_MODEL = True
DATA_THREADS = 1
DTYPE=tf.float32
RANK = 0
RANKS = 1
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
    avg_acc = tf.cast(acc,tf.float32)
    avg_sol = tf.cast(solved,tf.float32)
    avg_loss = tf.cast(loss,tf.float32)
    avg_acc_op = hvd.allreduce(avg_acc)
    avg_sol_op = hvd.allreduce(avg_sol)
    avg_loss_op = hvd.allreduce(avg_loss)
    acc = sess.run(avg_acc_op)
    solved = sess.run(avg_sol_op)
    loss = sess.run(avg_loss_op)
    return acc, solved, loss


def banner_print(string):
    if DEBUG:
        sep = "----------------------------------------"
        print("%s\n%s\n%s"%(sep,string,sep))


def compute_accuracy_class(target, output):
    tds = utils_np.graphs_tuple_to_data_dicts(target)
    solved = [ int(np.argmax(t["globals"]) == np.argmax(o)) for t,o in zip(tds,output[0]) ]
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
    data_threads = du.DataLoader(items,batch_size*4,nthreads=DATA_THREADS,shuf=shuffle)
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
        sys.stdout.write("    Batch x100:")
        sys.stdout.flush()
    for b, batch in enumerate(batch_generator()):
        items, input_dicts_p, input_dicts_l, target_dicts = batch
        # Progress print.
        if DEBUG and b % 100 == 0:
            sys.stdout.write(".")
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
    global hvd
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
    print(args)
    if args.hvd:
        print("Starting horovod...")
        import horovod.tensorflow as hvd_temp
        hvd = hvd_temp
        hvd.init()
        RANK = hvd.rank()
        RANKS = hvd.size()
        print("Initialization of horovod complete...")
        if(args.inference_out != None):
            INFERENCE_OUT = str(args.inference_out).split(".")[0] + "_%s.map"%str(RANK)
            print("Rank %s"%str(RANK), " is saving inference output to %s"%str(INFERENCE_OUT))
        
    if RANK != 0:
        DEBUG = False
    banner_print("PharML-GNN")
    if DEBUG:
        print(args)
    # Return parsed args.
    return args


def load_map_items(args):
    # Load data and build needed train / test datasets.
    banner_print("Loading data.")
    train_items = du.load_map_file(args.map_train, RANK, RANKS, debug=DEBUG)
    test_items = du.load_map_file(args.map_test, RANK, RANKS, debug=DEBUG)
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
    
    #tf.initialize_all_variables()
    #Save the model file
    #tf.saved_model.simple_save(sess, restore_path, inputs=)
    #model_save_path="./checkpoints/"
    #if QUANTIZE_MODEL:
        #ops={"input_p": input_p_op, "input_l": input_l_op, "target": target_op, "loss": loss_op, "outputs": output_ops}
        #if RANK==0:
        #    print("Saving model file...")
        #    if tf.get_default_session() != None:
        #        session = tf.get_default_session()
        #    else:   
        #        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        #        config.gpu_options.allow_growth = True
        #        config.gpu_options.visible_device_list = '0'
        #        session = tf.Session(config=config)

        #    tf.compat.v1.saved_model.simple_save(session, model_save_path, inputs={"input_p": inputs_p_op[0], "input_l": inputs_l_op[0]}, outputs={"outputs": output_ops[0]})
        #    #model = tf.keras.models.Model()
        #    print("Success, model file saved.")
 

    # Return ops and placeholders
    return inputs_p_ph, inputs_l_ph, targets_ph, inputs_p_op, inputs_l_op, targets_op, output_ops, loss_op


def build_keras_gnn(args,input_shapes):
    # Setup the GNN model.
    banner_print("Building keras model.")
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

    print("PharML-v2.0:")
    print(" -> tf.keras version:", tf.keras.__version__)
    print(" -> tf version: ", tf.__version__)
    print(" -> sonnet version: ", snt.__version__)
    
    
    merged_output = tf.concat([output_p_ops[0].globals,output_l_ops[0].globals],axis=-1)
    input_layer=tf.keras.Input(shape=(merged_output.shape[0]), batch_size=BATCH_SIZE, sparse=True, tensor=merged_output)
    classifier0 = tf.keras.layers.DenseFeatures(num_features_p*num_features_l, activation=tf.nn.relu)(input_layer)

#    initers = { "w": tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
#                "b": tf.truncated_normal_initializer(stddev=1.0)}
#    regs = { "w": tf.contrib.layers.l1_regularizer(scale=0.1),
#             "b": tf.contrib.layers.l2_regularizer(scale=0.1)}
#    classifier0 = snt.Linear(num_features_p*num_features_l, initializers=initers, regularizers=regs)
#    classifier1 = snt.Linear(num_features_p+num_features_l,initializers=initers, regularizers=regs)
#    final_output = snt.Linear(2, initializers=initers, regularizers=regs)
    #classifier0 = Dense(num_features_p*num_features_l)(input_layer)
    
    classifier1 = tf.keras.Dropout(Dense(int(num_features_p+num_features_l,activation=tf.nn.relu)))(classifier0)
    final_output = Dense(2,activation=tf.nn.softmax)(classifier1)
    model = tf.keras.Model(inputs=input_layer, outputs=final_output)
    print("keras model summary: ", model.summary())
    output_ops = [final_output(tf.nn.relu(classifier1(tf.nn.relu(tf.nn.dropout(classifier0(tf.nn.relu(merged_output)),keep_prob=0.5)))))]
    if MODE == 'classification':
        loss_ops = [ tf.losses.softmax_cross_entropy(targets_ph.globals, output_ops[0]) ]
    elif MODE == 'regression':
        loss_ops = [ tf.losses.mean_squared_error(targets_ph.globals, output_ops[0]) ]
    loss_op = sum(loss_ops)



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
        learning_rate = clr.cyclic_learning_rate(global_step=global_step, learning_rate=lr_decay, max_lr=1000*lr_decay,
                                                 step_size=2*step_sz, mode='triangular', gamma=.999)

    else:
        learning_rate = lr_init
        if RANK == 0:
            print("Using constant LR: ", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate) 
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    if args.hvd:
        optimizer = hvd.DistributedOptimizer(optimizer, use_locking=False)
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
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        local_path=str(os.getcwd())
        checkpoint_dir = local_path+'/checkpoints' if RANK == 0 else './checkpoints_test'
    else:            
        local_path=str(os.getcwd())
        config.gpu_options.visible_device_list = '0'
        checkpoint_dir = local_path+'/checkpoints'
    try:
        os.mkdir(checkpoint_dir)
    except OSError:  
        print ("Creation of directory %s failed!"%checkpoint_dir)
    else:  
        print ("Successfully created directory %s."%checkpoint_dir)
    sess = tf.Session(config=config)
    # Initialize Model.
    #print("Initializing global variables...")
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    #print("Done global variables init.")

    #saver = tf.train.Saver()
    
    model_path = checkpoint_dir + '/model%s.ckpt'%RANK
    restore_path = model_path
    
    #if RANK==0:
    #    # Test Save / Restore model with Rank 0.
    #    save_path = saver.save(sess, model_path)
    #    print("Coordinator Test checkpoint saved to: %s"%save_path)
    #else:
    #    save_path = saver.save(sess, model_path)
    #    print("Worker test checkpoint saved to: %s"%save_path)

        

    
    
    if args.restore != None:
        restore_path = args.restore
        print("Restoring model from: %s"%restore_path)
        saver = tf.compat.v1.train.import_meta_graph(restore_path+".meta",clear_devices=True)
        print("Got meta graph...")
        saver.restore(sess,restore_path)
        
        #tf.initialize_all_variables()
        print("Model restored sucessfully.")
        path_pb = str('/lus/sonexion/jbalma/temp/quantized_check_model0-test_bindingdb_2019m4_1of75pct-np-1-lr0.00000001-5,5-layer-32,32x2,2-bs_16-epochs-1000-nf-16,16/checkpoints/')
        path_qm = str(checkpoint_dir+'/tflite_models/')
        meta_path = str(restore_path)+'.meta'
        if QUANTIZE_MODEL:
            # Get outputs:
            # Restore the graph
            #saver_graph = tf.train.import_meta_graph(meta_path)
            #saver_graph = tf.compat.v1.train.import_meta_graph(meta_path)
            # Load weights
            #saver_graph.restore(sess,tf.train.latest_checkpoint(restore_path))

            print("Re-Initializing global variables...")
            #sess.run(tf.global_variables_initializer())
            #sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print("Re-Done global variables init.")

            # Get names
            #input_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            #output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            #output_node_names = ['Const']
            #output_node_names = []
            #for n in tf.get_default_graph().as_graph_def().node:
            #    #if(("save" not in n.name) and ("Assign" not in n.name)):
            #    output_node_names.append(n.name)
                
            print("output_node_names=",output_node_names)
            # Freeze the graph
            #frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            #            sess,
            #            sess.graph_def,
            #            output_node_names)

            converted_graph_def = create_inference_graph(
                input_saved_model_dir=restore_path,
                minimum_segment_size=3,
                is_dynamic_op=True,
                maximum_cached_engines=1)
            print("Successful conversion...")

            #frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            #    sess,
            #    tf.get_default_graph().as_graph_def(),
            #    output_node_names=['softmax_cross_entropy_loss'])

                # Save the frozen graph
            #with open(path_pb, 'wb') as f:
            #with tf.gfile.GFile(path_pb+'frozen_model.pb',"wb") as f:
            #    f.write(frozen_graph_def.SerializeToString())
            tf.io.write_graph(converted_graph_def, './', 'frozen_model.pb', as_text=False)

            #print("Re-Initializing global variables...")
            #sess.run(tf.global_variables_initializer())
            #sess.run(tf.local_variables_initializer())
            #print("Re-Done global variables init.")
            #import tensorflow.contrib.tensorrt as trt
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            #from tensorflow.python.compiler.tensorrt import trt_convert as trt
            if(0):
                print("nodes in frozen_graph_def:")
                for node in frozen_graph_def.node:
                    print(node.op)

                  # for fixing the bug of batch norm
                #gd = sess.graph.as_graph_def()
                gd = frozen_graph_def
                for node in gd.node:            
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in xrange(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] = node.input[index] + '/read'
                    elif node.op == 'AssignSub':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr: del node.attr['use_locking']
                    elif node.op == 'AssignAdd':
                        node.op = 'Add'
                        if 'use_locking' in node.attr: del node.attr['use_locking']
                    #elif node.op == 'Assign':
                    #    node.op = 'AssignFloat'
                    #    if 'use_locking' in node.attr: del node.attr['use_locking']
                    #    if 'T' in node.attr: node.attr['T']='DT_FLOAT32'
                    #    if 'validate_shape' in node.attr: del node.attr['validate_shape']

                converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, gd, output_node_names)
                #tf.train.write_graph(converted_graph_def, './', 'quant_model', as_text=False)
                tf.io.write_graph(converted_graph_def, './', 'quant_model', as_text=False)
                converter = trt.TrtGraphConverter(input_saved_model_dir=path_pb+'output_model.pb',#input_graph_def=converted_graph_def
                            session_config=config, max_batch_size=args.batch_size_test, is_dynamic_op=True, precision_mode="FP16",
                            nodes_blacklist=['logits', 'linear_0','w'])
                trt_graph = converter.convert()
                output_node = tf.import_graph_def(
                            trt_graph,
                            return_elements=['logits', 'linear_0','w'])
                print("running converted model")
                sess.run(output_node)

            else:
                converter = trt.TrtGraphConverter(input_saved_model_dir=path_pb+'frozen_model.pb')
                converter.convert()
                converter.save('./quantized_model')

                input_p_feed_dict = utils_tf.get_feed_dict(inputs_p_ph, inputs_p_op)
                input_l_feed_dict = utils_tf.get_feed_dict(inputs_l_ph, inputs_l_op)
                #target_feed_dict = utils_tf.get_feed_dict(target_ph, target_graphs)
                feed_dict = dict()
                feed_dict.update(input_p_feed_dict)
                feed_dict.update(input_l_feed_dict)
                #feed_dict.update(target_feed_dict)
                # Run it.
                #input_ops = { "input_p": inputs_p_op.run(), "input_l": inputs_l_op.run()} #, "target": target_op, "loss": loss_op, "outputs": output_ops }
                #if step_op != None:
                #    ops["step"] = step_op
                #output_tensors =
                #run_values = session.run(ops, feed_dict=feed_dict)
                #input_tensor = sess.run(input_ops, feed_dict) #, sess.run(inputs_l_op, {"input_l":inputs_l_ph})] #{"input_p":inputs_p_op, "input_l": inputs_l_op}
                #output_tensors = ["output"]
                output_tensor = [sess.run(output_ops)] #{"output":output_ops}


                # First load the SavedModel into the session    
                tf.saved_model.loader.load(
                    sess, [tf.saved_model.tag_constants.SERVING], './')
                output = sess.run([output_tensor], feed_dict=feed_dict)
            #converted _graph_def = trt.create_inference_graph(
            #input_graph_def = frozen_graph,
            #output_node_names-[‘logits’, ‘classes’])
            #params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            #precision_mode='FP16')
            #converter = trt.TrtGraphConverter(
            #input_saved_model_dir=input_saved_model_dir, conversion_params=params)
            #converter.convert()
            #converter.save('./')

            
            if(0):
                #model_path=path_pb+'output_model.pb'
                #meta_path=path_pb+'model0.ckpt.meta'
                #ckpt_path=path_pb+'model0.ckpt'
                #gsaver = tf.train.import_meta_graph(meta_path, clear_devices=True)
                #gsaver.restore(sess, tf.train.latest_checkpoint(model_path))
                #sess.run(tf.global_variables_initializer())
                #sess.run(tf.local_variables_initializer())
                #saver.restore(sess, tf.train.latest_checkpoint(path_pb))
                #gsaver.restore(sess, tf.train.latest_checkpoint(model_path))
        
                #output_node_names = 'embeddings'
        
                # for fixing the bug of batch norm
                gd = sess.graph.as_graph_def()
                #gd = frozen_graph_def
                for node in gd.node:            
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in xrange(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] = node.input[index] + '/read'
                            #if 'float_def' in node.input[index]:
                            #    node.input[index] = node.input[index].split('_')[0]+ 'float' + '/read'
                    elif node.op == 'AssignSub':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr: del node.attr['use_locking']
                    elif node.op == 'AssignAdd':
                        node.op = 'Add'
                        if 'use_locking' in node.attr: del node.attr['use_locking']
#                    elif node.op == 'Assign':
#                        node.op = 'AssignFloat32'
                        

        
                converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, gd, output_node_names)
                tf.train.write_graph(converted_graph_def, './', 'fixed_model.pb', as_text=False)

                # Save the frozen graph
                #import tensorflow as tf
                #converter = tf.lite.TFLiteConverter.from_saved_model(path_pb)
                #print("converter created...")
                #converter = lite.TFLiteConverter.from_frozen_graph(path_pb,  
                #inputs_p_op, inputs_l_op, targets_op, output_ops
                #inputs_p_op.name="input_p"
                #inputs_l_op.name="input_l"
            
                #input_tensors = ["input_p","input_l"]
                # Build a feed dict for the data.
                #inputs_p_op, inputs_l_op
                input_p_feed_dict = utils_tf.get_feed_dict(inputs_p_ph, inputs_p_op)
                input_l_feed_dict = utils_tf.get_feed_dict(inputs_l_ph, inputs_l_op)
                #target_feed_dict = utils_tf.get_feed_dict(target_ph, target_graphs)
                feed_dict = dict()
                feed_dict.update(input_p_feed_dict)
                feed_dict.update(input_l_feed_dict)
                #feed_dict.update(target_feed_dict)
                # Run it.
                input_ops = { "input_p": inputs_p_op.run(), "input_l": inputs_l_op.run()} #, "target": target_op, "loss": loss_op, "outputs": output_ops }
                #if step_op != None:
                #    ops["step"] = step_op
                #output_tensors = 
                #run_values = session.run(ops, feed_dict=feed_dict)
                input_tensors = sess.run(input_ops, feed_dict) #, sess.run(inputs_l_op, {"input_l":inputs_l_ph})] #{"input_p":inputs_p_op, "input_l": inputs_l_op}
                #output_tensors = ["output"]
                output_tensors = [sess.run(output_ops)] #{"output":output_ops}
                converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors, output_tensors)
                #converter = tf.lite.TFLiteConverter.from_frozen_graph('./fixed_model.pb', input_tensors, output_tensors)
                tflite_model = converter.convert()
                #converter = tf.lite.TFLiteConverter.from_saved_model(path_pb, input_tensors, output_tensors)
                open("converted_model.tflite", "wb").write(tflite_model)
                #converter = tf.lite.TFLiteConverter.from_frozen_graph(, input_tensors, output_tensors)
                #converter = tf.compat.v1.lite.TFLiteConverter.
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
                tflite_quant_model = converter.convert()
                print("Model restored sucessfully, Original Size:")
                tflite_models_dir = pathlib.Path(path_qm)
                tflite_models_dir.mkdir(exist_ok=True, parents=True)
                tflite_model_file = tflite_models_dir/"model0.tflite"
                tflite_model_file.write_bytes(tflite_quant_model)
                #Done with convert and write of original model
                #Quantize it, print out new size and save it
                print("Model Quantized sucessfully, Quantized Size:")
                tf.logging.set_verbosity(tf.logging.INFO)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
                tflite_fp16_model = converter.convert()
                tflite_model_fp16_file = tflite_models_dir/"model0_quant_f16.tflite"
                tflite_model_fp16_file.write_bytes(tflite_fp16_model)
                print("Quantized model saved sucessfully")
            
            
            if RANK==0:
                # Test Save / Restore model with Rank 0.
                save_path = saver.save(sess, model_path)
                print("Coordinator checkpoint of QUANTIZED model saved to: %s"%save_path)

        
        print("To resume training use --restore %s"%str(os.getcwd()+"/"+restore_path))
    else:
        restore_path=model_path
        saver.restore(sess, restore_path)
        print("Worker fresh checkpoint restore test success.")
        print("To resume training use --restore %s"%str(os.getcwd()+"/"+restore_path))
        print("Training new model.")

    
    if args.hvd:
        print("Broadcasting...")
        bcast_op = hvd.broadcast_global_variables(0)
        sess.run(bcast_op)
        print("Done broadcast")


    # Print total model parameters.
    if DEBUG:
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total trainable params: ", total_parameters)

    # Convert to quantized model for inference
    if INFERENCE_ONLY:
        banner_print("Start test loop for Inference-Only mode.")
        if QUANTIZE_MODEL:
            banner_print("Using Quantized Model!")
    else:    
        banner_print("Start training / testing loop.")

    # Training / inference loop.
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
        if acc > acc_best:
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
            break;

        #If test accuracy has not improved for more than 15 epochs, call it converged and exit
        if( (epoch-epoch_best) >= 15):
            print("Model Converged! Exiting Nicely...")
            #sys.exit(0)
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
    protein, ligand, target = du.load_item(test[0])
    # Build the GNN model.
    ops = build_gnn(args,(protein,ligand,target))
    # Setup the training optimizer
    if INFERENCE_ONLY:
        optimizer = None
        ops += (None,)
    else:
        optimizer, step_op = build_optimizer(args,ops[-1],len(train))
        ops += (step_op,)
    # Run the training / inference loop.
    run_gnn(args, ops, test, train_items=train, optimizer=optimizer)

    
if __name__== "__main__":
    main()


############################################################
