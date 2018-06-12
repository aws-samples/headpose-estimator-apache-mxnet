from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import resnet_model_headpose
import pickle
import numpy as np
import urllib
import sys

from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"

HEIGHT = 84
WIDTH = 84
DEPTH = 3
NUM_CLASSES = 9
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
RESNET_SIZE = 50
BATCH_SIZE = 200

# Scale the learning rate linearly with the batch size. When the batch size is
# 128, the learning rate should be 0.1.
_INITIAL_LEARNING_RATE = 0.1 * BATCH_SIZE / 128
_MOMENTUM = 0.9

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4

_BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE

def model_fn(features, labels, mode, params):
    """Model function for HeadPose ResNet50."""

    inputs = features[INPUT_TENSOR_NAME]
    tf.summary.image('images', inputs, max_outputs=6)
    
    network = resnet_model_headpose.resnet_v2(RESNET_SIZE, NUM_CLASSES)
    network.default_image_size = HEIGHT
    
    logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
    
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=tf.one_hot(labels, NUM_CLASSES)) 

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        boundaries = [int(_BATCHES_PER_EPOCH * epoch) for epoch in [100, 150, 200]]
        values = [_INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(tf.one_hot(labels, NUM_CLASSES), axis=1), predictions['classes']) # labels -> tf.one_hot(labels, NUM_CLASSES)
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def serving_input_fn(params):
    inputs = tf.convert_to_tensor( X_test )
    return build_raw_serving_input_receiver_fn( {INPUT_TENSOR_NAME: inputs} 
        )()

def train_input_fn(training_dir, params):
    global X_train, X_test, y_train, y_test  # share the data globally 
    trn_im, test_im, trn_output, test_output = load_data(training_dir)

    '''
    # Additional Data Augmentation #
    '''
    # Mirroring the headpose data
    trn_im_mirror = trn_im[:,:,:,::-1]
    trn_output_mirror = np.zeros(trn_output.shape)
    trn_output_mirror[:,0] = trn_output[:,0] 
    trn_output_mirror[:,1] = trn_output[:,1] * -1
    trn_im = np.concatenate((trn_im, trn_im_mirror), axis = 0) 
    trn_output = np.concatenate((trn_output, trn_output_mirror), axis = 0) 
        
    '''
    # Head Pose Labeling #
    '''
    # angle class (3) i.e. headpose class ( 3 x 3)
    n_grid = 3
    angles_thrshld = [np.arcsin(float(a) * 2 / n_grid - 1)/np.pi * 180 / 90 for a in range(1,n_grid)]
    
    # From (normalized) angle to angle class
    trn_tilt_cls = []
    trn_pan_cls = []
    for i0 in range(trn_output.shape[0]):
        trn_tilt_cls += [angles2Cat(angles_thrshld, trn_output[i0,0])]
        trn_pan_cls += [angles2Cat(angles_thrshld, trn_output[i0,1])]

    test_tilt_cls = []
    test_pan_cls = []
    for i0 in range(test_output.shape[0]):
        test_tilt_cls += [angles2Cat(angles_thrshld, test_output[i0,0])]
        test_pan_cls += [angles2Cat(angles_thrshld, test_output[i0,1])]
    
    np_trn_tilt_cls = np.asarray(trn_tilt_cls)
    np_test_tilt_cls = np.asarray(test_tilt_cls)
    np_trn_pan_cls = np.asarray(trn_pan_cls)
    np_test_pan_cls = np.asarray(test_pan_cls)
    
    # From angle class to head pose class
    np_trn_grid_cls = np_trn_pan_cls * n_grid + np_trn_tilt_cls
    np_test_grid_cls = np_test_pan_cls * n_grid + np_test_tilt_cls
    
    # trn_im: N C H W
    # test_im: N C H W
    ## TF ver. 1.4 only supports NHWC
    
    trn_im = np.swapaxes(trn_im, 1, 3)
    X_train = np.swapaxes(trn_im, 1, 2)
        
    test_im = np.swapaxes(test_im, 1, 3)
    X_test = np.swapaxes(test_im, 1, 2)
    
    y_train = np_trn_grid_cls
    y_test = np_test_grid_cls
    
    return tf.estimator.inputs.numpy_input_fn(
        x = {INPUT_TENSOR_NAME: np.array( X_train ) },
        y = np.array(y_train),
        num_epochs = 60,
        shuffle = True )()

def load_data(path):
    ### #Aspect Ratio 1:1 # 6.7 GB
    trn_im, test_im, trn_output, test_output = pickle.load(open(find_file(path,"HeadPoseData_trn_test_x15_py2.pkl"), 'rb'))
    return trn_im, test_im, trn_output, test_output

def find_file(root_path, file_name):
    for root, dirs, files in os.walk(root_path):
        if file_name in files:
            return os.path.join(root, file_name)
        
def angles2Cat(angles_thrshld, angl_input):
    # angl_input: Normalized angle -90 - 90 -> -1.0 - 1.0
    angles_cat_temp = angles_thrshld + [angl_input]
    return np.argmin(np.multiply(sorted(angles_cat_temp)-angl_input,sorted(angles_cat_temp)-angl_input))


def eval_input_fn(training_dir, params):
    # The evaluation data were pre-processed in the train_input_fn()
    # and made available as global variables which we reuse here
    # return a function to feed test data into predictions
    return tf.estimator.inputs.numpy_input_fn(
        x = {INPUT_TENSOR_NAME: np.array( X_test ) },
        y= np.array(y_test),
        num_epochs = None,
        shuffle = False )()
