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
import networkx as nx


############################################################


NUM_LAYERS  = 2   # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.
NUM_FILTERS = 64


############################################################

  
def conv_build(inputs):
  print("conv_build: input tensor shape: ", inputs.get_shape())
  ins = snt.BatchReshape(shape=(-1,1))(inputs)
  outputs = snt.Conv1D(output_channels=NUM_FILTERS, kernel_shape=1, rate=1)(ins)
  #outputs = tf.nn.relu(outputs)
  #outputs = snt.Conv1D(output_channels=NUM_FILTERS, kernel_shape=1, rate=1)(outputs)
  outputs = snt.LayerNorm()(outputs)
  mod = snt.MergeDims(start=1, size=2)
  outputs = mod(outputs)
  outputs = snt.Linear(output_size=LATENT_SIZE)(outputs)
  #outputs = tf.nn.relu(outputs)
  print("conv_build: USING CONV1D with reshaped inputs shape  =", ins.get_shape(), ", output shape: ", outputs.get_shape())
  tf.logging.info("Instantiated custom Conv1D layer model")
  return outputs


def make_conv_model():
  base_merge = snt.Module(conv_build, name='conv_layer')
  combined_model = snt.Sequential([base_merge, snt.LayerNorm()])
  tf.logging.info("Instantiated conv layer model + MLP")
  print("make_conv_model: USING CONV")
  return combined_model


def make_mlp_model():
  regs = { "w":tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)}
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True, use_dropout=False, initializers=regs),
      snt.LayerNorm()
  ])

def make_out_model():
  regs = { "w":tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)}

  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True, use_dropout=False, initializers=regs),
      snt.LayerNorm()
  ])

############################################################


class MLPGraphIndependent(snt.AbstractModule):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=make_mlp_model,
          node_model_fn=make_mlp_model,
          global_model_fn=make_mlp_model)

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphIndependentOut(snt.AbstractModule):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependentOut"):
    super(MLPGraphIndependentOut, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=make_out_model,
          node_model_fn=make_out_model,
          global_model_fn=make_out_model)

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(make_mlp_model, make_mlp_model,
                                           make_mlp_model)

  def _build(self, inputs):
    return self._network(inputs)


class ConvGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="ConvGraphNetwork"):
    super(ConvGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(make_conv_model, make_conv_model,
                                           make_conv_model)

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
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._num_processing_steps = num_processing_steps
    self._encoder = MLPGraphIndependent()
    self._cores = []
    for step in range(self._num_processing_steps):
      self._cores.append(MLPGraphNetwork(name=name+"_core%d"%step))
    self._decoder = MLPGraphIndependentOut()
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
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for step,core in enumerate(self._cores):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = core(core_input)
    decoded_op = self._decoder(latent)
    output_ops.append(self._output_transform(decoded_op))
    return output_ops


############################################################
