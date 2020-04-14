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
# Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
#           |         |---->|      |     |         |
#           *---------*     *------*     *---------*
#
############################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
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
DEBUG=False
MODE='classification'
INFERENCE_ONLY=False
DATA_THREADS=1
############################################################


def plot_history(x_vals, lst_train, lst_test, plot_label):
    #Summary and Plots
    plt.clf()
    plt.grid(True)
    plt.plot(x_vals,lst_train)
    plt.plot(x_vals,lst_test)
    plt.title(plot_label)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plot_name = './'+plot_label+'.svg'
    plt.savefig(plot_name, bbox_inches='tight')


def average_distributed_metrics(solved, count, loss):
    #Average over workers metrics when using horovod
    avg_acc = tf.cast(solved,tf.float32)
    avg_cor = tf.cast(count,tf.float32)
    avg_loss = tf.cast(loss,tf.float32)
    avg_acc_op = hvd.allreduce(avg_acc)
    avg_cor_op = hvd.allreduce(avg_cor)
    avg_loss_op = hvd.allreduce(avg_loss)
    solved = sess.run(avg_acc_op)
    count = sess.run(avg_cor_op)
    loss = sess.run(avg_loss_op)
    return solved, count, loss


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


def item_batch_iter(items, batch_size):
    # Create data loading threads for the set of items.
    data_threads = du.DataLoader(items,batch_size*4,nthreads=DATA_THREADS)
    data_threads.start()
    # Init batch of input / target items.
    pd, ld, td = ([],[],[])
    # Process all items. 
    elapsed = 0.0
    for item in range(len(items)):
        # If full batch, yield it and reset batch.
        if len(pd) == batch_size:
            yield (pd, ld, td)
            pd, ld, td = ([],[],[])
        # Add new item to current batch.
        start_time = time.time()
        p,l,t = data_threads.read_item()
        pd.append(p)
        ld.append(l)
        td.append(t)
        elapsed += time.time() - start_time
    # Yield the last (partial) batch.
    yield (pd, ld, td)
    # Join with data threads.
    data_threads.join()
    if DEBUG:
        print("\n    Wait:  %.2fs"%elapsed)


def run_batches(session, batch_size, input_p_ph, input_l_ph, target_ph, input_p_op, input_l_op, target_op, output_ops, step_op, loss_op, items):
    # Init counters / stats.
    start_time = time.time()
    solved, count, loss = (0.0, 0.0, 0.0)
    # Process data in batches.
    if DEBUG:
        num_batches = int(len(items)/batch_size)
        sys.stdout.write("    Number data threads %d," % DATA_THREADS)
        sys.stdout.write("    Required batches %d\n" % num_batches)
        sys.stdout.write("    Batch x100:")
        sys.stdout.flush()
    for b, batch in enumerate(item_batch_iter(items,batch_size)):
        input_dicts_p, input_dicts_l, target_dicts = batch
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
        if MODE=='classification':
            s, c = compute_accuracy_class(run_values["target"], run_values["outputs"])
        if MODE=='regression':
            s, c = compute_accuracy_reg(run_values["target"], run_values["outputs"])
        solved, count, loss = (solved+s, count+c, loss+run_values["loss"])
    elapsed = time.time() - start_time
    # Return stats.
    return elapsed, solved, loss, count


############################################################


# Parse command line args.
parser = argparse.ArgumentParser(prog='ML-Dock-GN using Tensorflow + graph_nets Backend', description='Processing input flags for Training Run.')
parser.add_argument('--batch_size', type=int, default=32, help='The (local) minibatch size.')
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
parser.add_argument('--inference_only', type=bool, default=False, help='Ignore training data and only use test set to make predictions, default False')
parser.add_argument('--data_threads', type=int, default=1, help='Number of threads used to buffer data into batches')
parser.add_argument('--mode', type=str, default="regression", help='Training mode: "regression" or "classification".')
parser.add_argument('--restore', type=str, default=None, help='Path to checkpoint file.')
args = parser.parse_args()
# Init horovod.
DEBUG          = args.debug
MODE           = args.mode
INFERENCE_ONLY = args.inference_only
DATA_THREADS   = args.data_threads

rank = 0
ranks = 1
if args.hvd:
    import horovod.tensorflow as hvd
    hvd.init()
    rank = hvd.rank()
    ranks = hvd.size()
    if rank != 0:
        DEBUG = False
banner_print("MLDock-GNN")
if DEBUG:
    print(args)
# Parse P/L options.
num_features = args.num_features.strip().split(",")
num_features_p, num_features_l = int(num_features[0]), int(num_features[1])
gnn_layers = args.gnn_layers.strip().split(",")
gnn_layers_p, gnn_layers_l = int(gnn_layers[0]), int(gnn_layers[1])
mlp_latent = args.mlp_latent.strip().split(",")
mlp_latent_p, mlp_latent_l = int(mlp_latent[0]), int(mlp_latent[1])
mlp_layers = args.mlp_layers.strip().split(",")
mlp_layers_p, mlp_layers_l = int(mlp_layers[0]), int(mlp_layers[1])
# Load data and build needed train / test datasets.
banner_print("Loading Data.")
batch_size = args.batch_size
train_items = du.load_map_file(args.map_train, rank, ranks, debug=True)
test_items = du.load_map_file(args.map_test, rank, ranks, debug=True)
# Optimizer.
tf.reset_default_graph()
banner_print("Building optimizer.")
global_step = tf.train.get_or_create_global_step()
# Learning Rate.
lr_init = float(np.sqrt(ranks)*args.lr_init)
if(args.use_clr):
    import clr
    if rank==0:
        print("Using Cyclic LR with initial LR: ", lr_init)
    step_sz = 10*(len(train_items)/batch_size)
    max_steps = args.epochs*ranks*(len(train_items)/batch_size)
    lr_decay = tf.train.exponential_decay(lr_init, global_step*ranks, max_steps, 0.5, staircase=True)
    learning_rate = clr.cyclic_learning_rate(global_step=global_step, learning_rate=lr_decay, max_lr=100*lr_decay,
                                             step_size=step_sz*2, mode='triangular2', gamma=.997)
else:
    learning_rate = lr_init
    if rank==0:
        print("Using constant LR: ", learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer=tf.train.AdamOptimizer(learning_rate)
if args.hvd:
    optimizer = hvd.DistributedOptimizer(optimizer)
    optimizer._learning_rate = tf.cast(learning_rate,tf.float32)
else:
    optimizer._learning_rate = tf.cast(learning_rate,tf.float32)
# Setup the GNN model.
banner_print("Building model.")
ip, il, it = du.load_item(train_items[0])
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
gnn_model_p = models.EncodeProcessDecode(edge_output_size=None, node_output_size=None, global_output_size=num_features_p,
                                         mlp_latent=mlp_latent_p, mlp_layers=mlp_layers_p,
                                         num_processing_steps=gnn_layers_p,
                                         name="gnn_model_protein")
gnn_model_l = models.EncodeProcessDecode(edge_output_size=None, node_output_size=None, global_output_size=num_features_l,
                                         mlp_latent=mlp_latent_l, mlp_layers=mlp_layers_l,
                                         num_processing_steps=gnn_layers_l,
                                         name="gnn_model_ligand")
# Setup loss function.
banner_print("Building loss function.")
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
if MODE=='classification':
    loss_ops = [ tf.losses.softmax_cross_entropy(targets_ph.globals, output_ops[0]) ]
if MODE=='regression':
    loss_ops = [ tf.losses.mean_squared_error(targets_ph.globals, output_ops[0]) ]
loss_op = sum(loss_ops)
step_op = optimizer.minimize(loss_op,global_step)
# Create new TF session.
banner_print("Create TF config / session.")
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
if args.hvd:
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    checkpoint_dir = './checkpoints' if rank == 0 else '/tmp/checkpoints'
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
sess.run(tf.global_variables_initializer())
# Sync Model parameters and report model size.
if args.hvd:
    bcast_op = hvd.broadcast_global_variables(0)
    sess.run(bcast_op)

# Save and Restore model.
saver = tf.train.Saver()
model_path = checkpoint_dir+'/model.ckpt'
save_path = saver.save(sess, model_path)
if DEBUG:
    print("Test checkpoint saved to: %s"%save_path)
restore_path = model_path
if args.restore != None:
    # Restore variables from disk.
    restore_path = args.restore
    print("Restoring model from: %s"%restore_path)
    saver.restore(sess,restore_path)
    print("Model restored sucessfully.")
else:
    print("Training new model.")
    saver.restore(sess, restore_path)
    if DEBUG:
        print("Fresh checkpoint restore test success.")
        print("To resume training use --restore %s"%str(os.getcwd()+"/"+restore_path))
total_parameters = 0
for variable in tf.trainable_variables():
    variable_parameters = 1
    for dim in variable.get_shape():
        variable_parameters *= dim.value
    total_parameters += variable_parameters
if DEBUG:
    print("Total trainable params: ", total_parameters)
# Training loop.
banner_print("Start training loop.")
acc_best = 0.0
logged_iterations = []
solveds_tr        = []
solveds_ge        = []
losses_tr         = []
losses_ge         = []
lr_hist           = []
for epoch in range(0, args.epochs):
    logged_iterations.append(epoch)
    if rank == 0:
        print("Epoch %d:"%(epoch))
        print("  Training.")
    # Run training step.
    if(INFERENCE_ONLY==False):
        elapsed, solved, loss, count = run_batches(sess, batch_size,
                                                   inputs_p_ph, inputs_l_ph, targets_ph,
                                                   inputs_p_op, inputs_l_op, targets_op,
                                                   output_ops, step_op, loss_op, train_items)
    else:
        if rank == 0:
            print("  Skipped Train, INFERENCE_ONLY enabled!")
        elapsed=0
        solved=0
        loss=1
        count=1

    acc = float(solved/count)
    loss = float(loss/count)
    lr = sess.run(optimizer._learning_rate)
    if args.hvd:
        acc, solved, loss = average_distributed_metrics(acc, solved, loss)
        count = hvd.size()*count
        solved = hvd.size()*solved
        if rank == 0:
            print("    LrnR:  %.6f"%lr)
    if rank == 0:
        print("    Time:  %.1fs"%(elapsed))
        print("    Loss:  %f"%(loss))
        print("    Acc.:  %f  (%.1f/%.1f)"%(acc,solved,count))
        print("  Testing.")
    solveds_tr.append(acc)
    losses_tr.append(loss)
    lr_hist.append(lr)

    # Run a test step.
    elapsed, solved, loss, count = run_batches(sess, batch_size*2,
                                               inputs_p_ph, inputs_l_ph, targets_ph,
                                               inputs_p_op, inputs_l_op, targets_op,
                                               output_ops, None, loss_op, test_items)
    acc = float(solved/count)
    loss = float(loss/count)
    if args.hvd:
        acc, solved, loss = average_distributed_metrics(acc, solved, loss)
        count = hvd.size()*count
        solved = hvd.size()*solved
    if rank == 0:
        print("    Time:  %.1fs"%(elapsed))
        print("    Loss:  %f"%(loss))
        print("    Acc.:  %f  (%.1f/%.1f)"%(acc,solved,count))
        # Checkpoint if needed.
        if acc > acc_best:
            acc_best = acc
            sys.stdout.write("  Checkpoint: ")
            sys.stdout.flush()
            save_path = saver.save(sess, model_path)
            print("%s"%(save_path))
    solveds_ge.append(acc)
    losses_ge.append(loss)
    if rank == 0:
        plot_history(logged_iterations, solveds_tr,solveds_ge, 'MLDock-Accuracy')
        plot_history(logged_iterations, losses_tr,losses_ge, 'MLDock-Loss')
        plot_history(logged_iterations, lr_hist,lr_hist, 'MLDock-LR')
# Success!
banner_print("Success!")


############################################################
