import argparse, math
from datetime import datetime
import h5py
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import tensorflow as tf
import tensorflow_probability as tfp
import socket, importlib, os, sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider, tf_util, aneurysm_dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#HYPERPARAMETERS:
FLAGS = {'GPU_INDEX' : 0,
         'MODEL_NAME' : 'pointnet2_wss_reg',  # name of file which stores the model
         'LOSS' : 'l1', # l1 or l2
         'METRICS' : ['l1', 'l2', 'avg_pearson'], # these scores are calculated for the validation sets
         'BATCH_SIZE' : 1,
         'NUM_POINT' : 1024, # number of points in each CA point cloud
         'FEATURES' : 4, # number of features for each point (must be 3 or more)
         'MAX_EPOCH' : 150,
         'BASE_LEARNING_RATE' : 1e-4,
         'OPTIMIZER' : 'adam', # adam, momentum, adagrad, adadelta, rmsprop or sgd
         'MOMENTUM' : None,
         'LR_DECAY_STEP' : 5000,  # learning rate = BASE_LEARNING_RATE * LR_DECAY_RATE ^ (SAMPLES_SEEN / LR_DECAY_STEP)
         'LR_DECAY_RATE' : 0.7,
         'BATCH_NORM' : False,
         'BN_INIT_DECAY' : 0.3, # starts from 1 - BN_INIT_DECAY
         'BN_DECAY_DECAY_RATE' : 0.2,
         'BN_DECAY_DECAY_STEP' : 800,
         'PATIENCE_ES' : 151, # patience for early stopping, < 0 for no early stopping
         'DROPOUT_RATE': 0.1,  # rate for dropout layer
         'MAX_POINT_DROPOUT_RATIO' : 0.0  # 0.0 for no point dropout
        }
EPOCH_CNT = 0 # counts the number of epochs
LEARNING_CURVE = defaultdict(list) # loss values stored here

#######
HOSTNAME = socket.gethostname()
MODEL = importlib.import_module(FLAGS['MODEL_NAME']) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS['MODEL_NAME']+'.py')
LOG_DIR = '../log'  # training items will be saved to this folder
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE.replace(' ', '\ '), LOG_DIR.replace(' ', '\ '))) # backup of model def
os.system('cp train_test.py %s' % (LOG_DIR.replace(' ', '\ '))) # backup of train procedure
os.system('cp aneurysm_dataset.py %s' % (LOG_DIR.replace(' ', '\ '))) # backup of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w') # open file to write output
LOG_FOUT.write(str(FLAGS) + '\n') # write hyperparameters to log file
MODEL_PATH = os.path.join(LOG_DIR, "model.ckpt") # the model is saved with this name 
#######

def log_string(out_str, printout=True):
  LOG_FOUT.write(out_str + '\n')
  LOG_FOUT.flush()
  if printout: print(out_str)

def get_learning_rate(batch):
  learning_rate = tf.compat.v1.train.exponential_decay(
                    FLAGS['BASE_LEARNING_RATE'], # Base learning rate.
                    batch * FLAGS['BATCH_SIZE'], # Current index into the dataset.
                    FLAGS['LR_DECAY_STEP'], # Decay step.
                    FLAGS['LR_DECAY_RATE'], # Decay rate.
                    staircase=False)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
  return learning_rate

def get_bn_decay(batch):
  bn_momentum = tf.compat.v1.train.exponential_decay(
                    FLAGS['BN_INIT_DECAY'], # Base batch norm decay.
                    batch * FLAGS['BATCH_SIZE'], # Current index into the dataset.
                    FLAGS['BN_DECAY_DECAY_STEP'], # Decay step.
                    FLAGS['BN_DECAY_DECAY_RATE'], # Decay rate.
                    staircase=True)
  bn_decay = tf.minimum(0.999, 1 - bn_momentum)
  return bn_decay

def get_optimizer(name, lr, **kwargs):
  if name == 'sgd':
    return tf.compat.v1.train.GradientDescentOptimizer(lr, **kwargs)
  elif name == 'momentum':
    return tf.compat.v1.train.MomentumOptimizer(lr, FLAGS['MOMENTUM'], **kwargs)
  elif name == 'adam':
    return tf.compat.v1.train.AdamOptimizer(lr, **kwargs)
  elif name == 'adagrad':
    return tf.compat.v1.train.AdagradOptimizer(lr, **kwargs)
  elif name == 'adadelta':
    return tf.compat.v1.train.AdadeltaOptimizer(lr, **kwargs)
  elif name == 'rmsprop':
    return tf.compat.v1.train.RMSPropOptimizer(lr, **kwargs)
  else:
    raise NotImplementedError('Unknown optimizer %s.' % str(name))

def get_metric_score(name, y_true, y_pred):
  if name in {'l1', 'mae', 'mean_absolute_error'}:
    return mean_absolute_error(y_true, y_pred)
  elif name in {'l2', 'mse', 'mean_squared_error'}:
    return mean_squared_error(y_true, y_pred)
  elif name in {'pearsonr', 'pearsonsr', 'pearson_r', 'pearsons_r'}:
    return [pearsonr(y_true[row,...], y_pred[row,...]) for row in range(y_true.shape[0])]
  elif name in {'avg_pearson', 'avg_pearsonr'}:
    corr_coefs = [pearsonr(y_true[row,...], y_pred[row,...])[0] for row in range(y_true.shape[0])]
    return np.mean(corr_coefs)
  elif name in {'r2', 'rsquared'}:
    return r2_score(y_true, y_pred)
  else: raise NotImplementedError('Unknown metric %s.' % str(name))
########

## TRAIN LOOP ##
def train(train_dataset, val_dataset, verbose=True):
  """
  train_dataset: Training dataset, object of class AneurysmDataset
  val_dataset: Validation dataset, object of class AneurysmDataset. Can be set to None.
  verbose: If set to False, this function does not print text to the console but still writes to the log file.
  """
  log_string('TRAINING SET SIZE: ' + str(len(train_dataset)), printout=verbose)
  if val_dataset is not None: log_string('VALIDATION SET SIZE: ' + str(len(val_dataset)), printout=verbose)

  tf.compat.v1.reset_default_graph()
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(FLAGS['GPU_INDEX'])):
      pointclouds_pl, labels_pl = MODEL.placeholder_inputs(FLAGS['BATCH_SIZE'],
                                                           FLAGS['NUM_POINT'],
                                                           FLAGS['FEATURES'])
      is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

      # Note the global_step=batch parameter to minimize. 
      # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
      batch = tf.compat.v1.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
      bn_decay = get_bn_decay(batch)
      tf.compat.v1.summary.scalar('bn_decay', bn_decay)

      # Get model and loss
      #pred, mid_xyz, mid_points = MODEL.get_model(pointclouds_pl, is_training_pl, batchnorm=FLAGS['BATCH_NORM'], bn_decay=bn_decay, dropout_rate=0.1)
      pred = MODEL.get_model(pointclouds_pl, is_training_pl, batchnorm=FLAGS['BATCH_NORM'], bn_decay=bn_decay, dropout_rate=FLAGS['DROPOUT_RATE'])
      MODEL.get_loss(pred, labels_pl, loss=FLAGS['LOSS'])
      losses = tf.get_collection('losses')
      total_loss = tf.add_n(losses, name='total_loss')
      tf.compat.v1.summary.scalar('total_loss', total_loss)
      for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)

      # Get training operator
      learning_rate = get_learning_rate(batch)
      tf.compat.v1.summary.scalar('learning_rate', learning_rate)
      optimizer = get_optimizer(name=FLAGS['OPTIMIZER'], lr=learning_rate)
      update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS) # for batchnorm
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step=batch)

      # Add ops to save and restore all the variables.
      saver = tf.compat.v1.train.Saver()

    # Create a session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.compat.v1.Session(config=config)

    # Add summary writers
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(os.path.join('log', 'train'), sess.graph)
    val_writer = tf.compat.v1.summary.FileWriter(os.path.join('log', 'val'), sess.graph)

    # Init variables
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    log_string('Parameters: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss,
           'train_op': train_op,
           'merged': merged,
           'step': batch}

    if len(train_dataset) % FLAGS['BATCH_SIZE'] != 0:
      log_string('WARNING: NUMBER OF SAMPLES NOT DIVISIBLE BY BATCH_SIZE!', printout=verbose)

    min_loss, early_stop_cnt = float('-inf'), 0
    for epoch in range(1, FLAGS['MAX_EPOCH'] + 1):
      log_string('** EPOCH %03d **' % (epoch), printout=verbose)
      sys.stdout.flush()
        
      train_one_epoch(sess, ops, train_writer, train_dataset, verbose=verbose)
      results = eval_one_epoch(sess, ops, val_writer, [(train_dataset, 'train'), (val_dataset, 'val')], verbose=verbose)
      for key, value in results.iteritems(): LEARNING_CURVE[key].append(value)

      # Early stopping
      if val_dataset is not None and FLAGS['PATIENCE_ES'] > 0:
        if results['val_avg_pearson'] >= min_loss:
          early_stop_cnt = 0
          min_loss = results['val_avg_pearson']
          save_path = saver.save(sess, MODEL_PATH)
          log_string("Model saved to file: %s" % save_path, printout=verbose)
        else:
          early_stop_cnt += 1
          if early_stop_cnt >= FLAGS['PATIENCE_ES']:
            early_stop_cnt = 0
            log_string('Early stopping at epoch %d' % epoch, printout=verbose)
            break
      elif epoch % 5 == 0:
          save_path = saver.save(sess, MODEL_PATH)
          log_string("Model saved in file: %s" % save_path, printout=verbose)


def get_batch(dataset, idxs, start_idx, end_idx):
  bsize = end_idx - start_idx
  batch_data = np.zeros((bsize, FLAGS['NUM_POINT'], FLAGS['FEATURES']), dtype=np.float32)
  batch_label = np.zeros((bsize, FLAGS['NUM_POINT']), dtype=np.float32)
  for i in range(bsize):
    ps, lbl, _ = dataset[idxs[start_idx + i]]
    batch_data[i, :, :] = ps
    batch_label[i, :] = lbl
  return batch_data, batch_label


def train_one_epoch(sess, ops, train_writer, dataset, verbose=True):
  """
  Train model for one epoch
  """
  global EPOCH_CNT
  is_training = True

  # Shuffle train samples
  train_idxs = np.arange(0, len(dataset))
  np.random.shuffle(train_idxs)

  num_batches = len(dataset) / FLAGS['BATCH_SIZE'] # discards samples if dataset not divisible by batch size

  log_string('[' + str(datetime.now()) + ' | EPOCH ' + str(EPOCH_CNT) + '] Starting training.', printout=False)

  loss_sum, batch_print_steps = 0, 10
  for batch_idx in range(num_batches):
    start_idx, end_idx = batch_idx * FLAGS['BATCH_SIZE'], (batch_idx + 1) * FLAGS['BATCH_SIZE']
    batch_data, batch_label = get_batch(dataset, train_idxs, start_idx, end_idx)
    # Perturb point clouds:
    batch_data[:,:,:3] = provider.jitter_point_cloud(batch_data[:,:,:3])
    batch_data[:,:,:3] = provider.rotate_perturbation_point_cloud(batch_data[:,:,:3])
    batch_data[:,:,:3] = provider.shift_point_cloud(batch_data[:,:,:3])
    batch_data[:,:,:3] = provider.random_point_dropout(batch_data[:,:,:3],
                                                       max_dropout_ratio=FLAGS['MAX_POINT_DROPOUT_RATIO'])
    feed_dict = {ops['pointclouds_pl']: batch_data,
                 ops['labels_pl']: batch_label,
                 ops['is_training_pl']: is_training}
    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'],
                                                     ops['loss'], ops['pred']], feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    loss_sum += loss_val
    if batch_idx % batch_print_steps == 0:
      log_string('[Batch %03d] Mean Loss: %f' % ((batch_idx + 1), (loss_sum / batch_print_steps)), printout=verbose)
      loss_sum = 0

def eval_one_epoch(sess, ops, val_writer, datasets_tuples, verbose=True):
  """
  Evaluate model for current epoch
  datasets_tuples: tuples to calculate metrics on, format: (dataset, name). 

  Returns: dictionary containing results. keys format is '{name}_{measure}' (eg. val_l1)
  """
  global EPOCH_CNT
  is_training = False

  results = {}

  log_string('[' + str(datetime.now()) + ' | EPOCH ' + str(EPOCH_CNT+1) + '] Starting evaluation.', printout=verbose)

  out_string = ''
  for dataset, name in datasets_tuples:
    if dataset is None: continue
    val_idxs = np.arange(0, len(dataset))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(dataset) + FLAGS['BATCH_SIZE'] - 1) / FLAGS['BATCH_SIZE']

    all_true = np.zeros((len(dataset), FLAGS['NUM_POINT'])).astype(np.float32) # stores true values for whole dataset
    all_pred = np.zeros((len(dataset), FLAGS['NUM_POINT'])).astype(np.float32) # stores predicted values for whole dataset

    batch_data = np.zeros((FLAGS['BATCH_SIZE'], FLAGS['NUM_POINT'], FLAGS['FEATURES'])).astype(np.float32)

    loss_sum = 0
    for batch_idx in range(num_batches):
      start_idx, end_idx = batch_idx * FLAGS['BATCH_SIZE'], min(len(dataset), (batch_idx + 1) * FLAGS['BATCH_SIZE'])
      cur_batch_data, cur_batch_label = get_batch(dataset, val_idxs, start_idx, end_idx)
      batch_data[0 : cur_batch_data.shape[0], :, :] = cur_batch_data

      feed_dict = {ops['pointclouds_pl']: batch_data,
                   ops['is_training_pl']: is_training}
      pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
      pred_val = pred_val[:cur_batch_data.shape[0], :].squeeze()

      all_true[start_idx:end_idx, :] = cur_batch_label
      all_pred[start_idx:end_idx, :] = pred_val

    for metric in FLAGS['METRICS']:
      results[name + '_' + metric] = get_metric_score(metric,  all_true, all_pred)
      if metric not in {'pearsonr', 'pearsonsr', 'pearson_r', 'pearsons_r'}:
        out_string += name + '_' + metric + ':' + str(round(results[name + '_' + metric], 8)) + (' '*2)

  log_string(out_string, printout=verbose)
  
  EPOCH_CNT += 1
  return results


def test(test_dataset):
  log_string('\nTESTING SET SIZE: ' + str(len(test_dataset)))

  tf.compat.v1.reset_default_graph()
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(FLAGS['GPU_INDEX'])):
      pointclouds_pl, labels_pl = MODEL.placeholder_inputs(len(test_dataset),
                                                           FLAGS['NUM_POINT'],
                                                           FLAGS['FEATURES'])
      is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

      # Get model and loss
      pred = MODEL.get_model(pointclouds_pl, is_training_pl, batchnorm=FLAGS['BATCH_NORM'])
      #pred, mid_xyz, mid_points = MODEL.get_model(pointclouds_pl, is_training_pl, batchnorm=FLAGS['BATCH_NORM'])
      MODEL.get_loss(pred, labels_pl, loss=FLAGS['LOSS'])
      losses = tf.get_collection('losses')
      total_loss = tf.add_n(losses, name='total_loss')
      saver = tf.compat.v1.train.Saver()

    # Create a session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.compat.v1.Session(config=config)
    
    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string('Model restored.')

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           #'mid_xyz': mid_xyz,
           #'mid_points': mid_points,
           'loss': total_loss}

    log_string('[' + str(datetime.now()) + ' | EPOCH ' + str(EPOCH_CNT) + '] Starting testing.', printout=False)

    feed_dict = {ops['pointclouds_pl']: test_dataset.point_sets,
                 ops['is_training_pl']: False}
    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
    pred_val = pred_val.squeeze(axis=-1)

    results = {}
    for metric in FLAGS['METRICS']:
      results['test_' + metric] = get_metric_score(metric, test_dataset.tawss_vals, pred_val)

  return pred_val, results#, mid_xyz, mid_points


if __name__ == '__main__':
  log_string('pid: %s'%(str(os.getpid())), printout=False)

  LEARNING_CURVE.clear()

  train_model = (sys.argv[1].lower() == 'true')

  FNAME_LIST_TRAIN = ['case_{}.txt'.format(str(x).zfill(2)) for x in list(range(1,34))]
  FNAME_LIST_VAL =  ['case_{}.txt'.format(str(x).zfill(2)) for x in list(range(34,38))]
  FNAME_LIST_TEST = ['case_{}.txt'.format(str(x).zfill(2)) for x in list(range(38,39))]

  DATA_PATH = os.path.join(ROOT_DIR, 'data', 'ca_data') # Path to the data set
  TRAIN_DATASET = aneurysm_dataset.AneurysmDataset(root=DATA_PATH,
                                                   npoints=FLAGS['NUM_POINT'],
                                                   fnames=FNAME_LIST_TRAIN)
  VAL_DATASET = aneurysm_dataset.AneurysmDataset(root=DATA_PATH,
                                                 npoints=FLAGS['NUM_POINT'],
                                                 fnames=FNAME_LIST_VAL,
                                                 max_norm=TRAIN_DATASET.max_norm,
                                                 wss_min_max=TRAIN_DATASET.wss_min_max)

  if train_model:
    train(TRAIN_DATASET, None, verbose=True)  # training procedure
   
  TEST_DATASET = aneurysm_dataset.AneurysmDataset(root=DATA_PATH,
                                                  npoints=FLAGS['NUM_POINT'],
                                                  fnames=FNAME_LIST_TEST,
                                                  max_norm=TRAIN_DATASET.max_norm,
                                                  wss_min_max=TRAIN_DATASET.wss_min_max)

  test_predictions, test_results = test(TEST_DATASET)  # testing procedure
  log_string('[Test set] - L1: {:.8f}  L2: {:.8f}  Avg. Pearson: {:.8f}'.format(\
                           get_metric_score('l1', TEST_DATASET.tawss_vals, test_predictions),
                           get_metric_score('l2', TEST_DATASET.tawss_vals, test_predictions),
                           get_metric_score('avg_pearson', TEST_DATASET.tawss_vals, test_predictions)
                           )
            )

  results = {'data': TEST_DATASET.point_sets,
             'y': TEST_DATASET.untransform(TEST_DATASET.tawss_vals),
             'y_hat': TEST_DATASET.untransform(test_predictions),
            }
  np.save(os.path.join(LOG_DIR, 'test_results.npy'), results)
  log_string('Saving test results to file: {}'.format(os.path.join(LOG_DIR, 'test_results.npy')))
  
  LOG_FOUT.close()
