# Copyright 2018 The GraphNets Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Model architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
import tensorflow as tf
import sonnet as snt


############################################################

  
def make_mlp_model(sizes):
  initers = { "w": tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
              "b": tf.truncated_normal_initializer(stddev=1.0)}
  regs    = { "w": tf.contrib.layers.l1_regularizer(scale=0.1),
              "b": tf.contrib.layers.l2_regularizer(scale=0.1)}

  return lambda: snt.Sequential([snt.nets.MLP(sizes,activate_final=True,use_dropout=False,initializers=initers, regularizers=regs),snt.LayerNorm()])


############################################################


class MLPGraphIndependent(snt.AbstractModule):
  def __init__(self, sizes, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=make_mlp_model(sizes),
          node_model_fn=make_mlp_model(sizes),
          global_model_fn=make_mlp_model(sizes))

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  def __init__(self, sizes, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(make_mlp_model(sizes),
                                           make_mlp_model(sizes),
                                           make_mlp_model(sizes))

  def _build(self, inputs):
    return self._network(inputs)


  
############################################################


class EncodeProcessDecode(snt.AbstractModule):
  """Full encode-process-decode model.

  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               num_processing_steps=2,
               mlp_layers=2,
               mlp_latent=16,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._num_processing_steps = num_processing_steps
    # One set of encoder weights.
    self._encoder = MLPGraphIndependent([mlp_latent]*mlp_layers)
    # One set of core weights per message passing step.
    self._cores = []
    for step in range(self._num_processing_steps):
      # Set a name
      step_name = name+"_core%d"%step
      # Number of layers and base latent size grows with depth.
      sizes = [mlp_latent+int((step*mlp_latent)/(self._num_processing_steps-1.0))]
      layers = mlp_layers + int((step/(self._num_processing_steps-1.0))*3.0)
      sizes = sizes * layers
      # Taper the MLP size list to add a choke point in the middle.
      for ndx,sz in enumerate(sizes):
        if ndx <= len(sizes)/2:
          sizes[ndx] = int(sz-(sz/1.33)*ndx/(len(sizes)-1.0))
        else:
          sizes[ndx] = int(sz-(sz/1.33)*((len(sizes)-1.0)-ndx)/(len(sizes)-1.0))
        if sizes[ndx] < 3:
          sizes[ndx] = 3
      # Verbose print the name and size
      print("%s: %s"%(step_name,str(sizes)))
      # Finally create the layer with the desired MLP shape.
      self._cores.append(MLPGraphNetwork(sizes,name=step_name))
    # One set of decoder weights.
    self._decoder = MLPGraphIndependent([mlp_latent]*mlp_layers)
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    with self._enter_variable_scope():
      self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn)

  def _build(self, input_op):
    # Initial step gets encoder output as input.
    latents = [ self._encoder(input_op) ]
    # Each sucsessive step gets all previous steps' outputs as input.
    for step,core in enumerate(self._cores):
      latents.append(core(utils_tf.concat(latents, axis=1)))
    # Return a single list of one output: decode of final core's output.
    return [ self._output_transform(self._decoder(latents[-1])) ]


############################################################
