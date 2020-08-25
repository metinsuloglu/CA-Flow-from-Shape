import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, num_feature):
  pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, num_feature))
  labels_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point))
  return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, batchnorm=False, bn_decay=None, dropout_rate=0.1):
  """
  PointNet++ for TAWSS regression, input is BxNxF, output BxN
  """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  num_features = point_cloud.get_shape()[2].value

  l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # point coordinates
  if num_features == 3: l0_points = None
  else: l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,1]) # scale information

  #mid_xyz = {'l0_xyz': l0_xyz}
  #mid_points = {'l0_points': l0_points}

   # Set Abstraction layers with multi-scale grouping
  l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 256, [0.1,0.2,0.4], [16,32,64], [[64,128], [128,128], [128,128]], is_training, bn_decay, dropout_rate, scope='layer1', bn=batchnorm)
  l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 16, [0.2,0.4,0.8], [16,32,64], [[128],[256],[512]], is_training, bn_decay, dropout_rate, scope='layer2', bn=batchnorm)
  l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3', bn=batchnorm)
  
  #mid_xyz['l2_xyz'] = l2_xyz
  #mid_points['l2_points'] = l2_points
  #mid_xyz['l1_xyz'] = l1_xyz
  #mid_points['l1_points'] = l1_points
  #mid_xyz['l3_xyz'] = l3_xyz
  #mid_points['l3_points'] = l3_points

  # Feature Propagation layers
  l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512], is_training, bn_decay, dropout_rate, scope='fp_layer1', bn=batchnorm)
  l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256], is_training, bn_decay, dropout_rate, scope='fp_layer2', bn=batchnorm)
  l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz, l0_points], axis=-1), l1_points, [128], is_training, bn_decay, dropout_rate, scope='fp_layer3', bn=batchnorm)

  # Fully Connected layers
  net = tf_util.conv1d(l0_points, 128, 1, scope='fc1', padding='VALID', is_training=is_training, bn=batchnorm, bn_decay=bn_decay)
  #mid_points['feats'] = net
  net = tf_util.dropout(net, rate=dropout_rate, is_training=is_training, scope='dp1')
  net = tf_util.conv1d(net, 1, 1, scope='fc2', padding='VALID', activation_fn=None, bn=False)

  return net#, mid_xyz, mid_points


def get_loss(pred, label, loss='l1'):
    """
    pred: BxN,
    label: BxN 
    """
    
    if loss in {'l1', 'mae', 'mean_absolute_error'}:
      reg_loss = tf.reduce_mean(tf.abs(tf.squeeze(label) - tf.squeeze(pred)))
    elif loss in {'l2', 'mse', 'mean_squared_error'}:
      reg_loss = tf.compat.v1.losses.mean_squared_error(labels=tf.squeeze(label), predictions=tf.squeeze(pred))
    elif loss in {'huber'}:
      reg_loss = tf.compat.v1.losses.huber_loss(labels=tf.squeeze(label), predictions=tf.squeeze(pred), delta=0.5)
    else: raise NotImplementedError('Unknown loss %s.' % str(loss))

    tf.compat.v1.summary.scalar(loss + ' loss', reg_loss)
    tf.compat.v1.add_to_collection('losses', reg_loss)
    return reg_loss
