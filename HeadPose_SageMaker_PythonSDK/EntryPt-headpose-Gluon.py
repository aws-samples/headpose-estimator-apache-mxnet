import logging

import mxnet as mx
import numpy as np
import os
import urllib
import pickle
import sys
import cv2
from mxnet import init
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.gluon.model_zoo.vision import resnet50_v1

### The shape of input image. 
### Aspect Ratio 1:1 # (1,3,84,84)
DSHAPE = (1,3,84,84)

##################################
###
### Helper functions
###
##################################

#cv2 is not supported in ml.m4 instance
def shiftHSV(im, h_shift_lim=(-180, 180),
                 s_shift_lim=(-255, 255),
                 v_shift_lim=(-255, 255), drop_p=0.5):
    if np.random.random() < drop_p:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(im)
        h_shift = np.random.uniform(h_shift_lim[0], h_shift_lim[1])
        h = cv2.add(h, h_shift) 
        s_shift = np.random.uniform(s_shift_lim[0], s_shift_lim[1])
        s = cv2.add(s, s_shift)
        v_shift = np.random.uniform(v_shift_lim[0], v_shift_lim[1])
        v = cv2.add(v, v_shift)
        im = cv2.merge((h, s, v))
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        im = np.uint8(im) 
        im = np.float32(im)
    return im                     
                                       
def load_data(path):
    ### #Aspect Ratio 1:1 # 6.7 GB
    trn_im, test_im, trn_output, test_output = pickle.load(open(find_file(path,"HeadPoseData_trn_test_x15_py2.pkl"), 'rb')) 

    print("dataset loaded !")
    return trn_im, test_im, trn_output, test_output

def find_file(root_path, file_name):
    for root, dirs, files in os.walk(root_path):
        if file_name in files:
            return os.path.join(root, file_name)

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)

def load_model(s_fname, p_fname):
    """
    Load model checkpoint from file.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    symbol = mx.symbol.load(s_fname)
    save_dict = mx.nd.load(p_fname)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params        
        

def angles2Cat(angles_thrshld, angl_input):
    # angl_input: Normalized angle -90 - 90 -> -1.0 - 1.0
    angles_cat_temp = angles_thrshld + [angl_input]
    return np.argmin(np.multiply(sorted(angles_cat_temp)-angl_input,sorted(angles_cat_temp)-angl_input))

# Accuracy Evaluation
def eval_acc(data_iter, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net(data)
        pred = nd.argmax(output, axis=1)
        acc.update(preds=pred, labels=label)
    return acc.get()[1]

# Training Loop
def train_util(output_data_dir, net, train_iter, validation_iter, loss_fn, trainer, ctx, epochs, batch_size):
    metric = mx.metric.create(['acc'])
    lst_val_acc = []
    lst_trn_acc = []
    best_accuracy = 0
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_iter):
            # ensure context            
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            
            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(data.shape[0])

        train_acc = eval_acc(train_iter, net, ctx)
        validation_acc = eval_acc(validation_iter, net, ctx)

        lst_trn_acc += [train_acc]
        lst_val_acc += [validation_acc]
        
        ### Modularize the output network. 
        # Export .json and .params files
        # chkpt-XX-symbol.json does not come with softmax layer on the top of network. 
        net.export('{}/chkpt-{}'.format(output_data_dir, epoch)) 
        # Overwrite .json with the one with softmax.
        net_with_softmax = net(mx.sym.var('data'))
        net_with_softmax = mx.sym.SoftmaxOutput(data=net_with_softmax, name="softmax")
        net_with_softmax.save('{}/chkpt-{}-symbol.json'.format(output_data_dir, epoch)) 
        print("Epoch %s | training_acc %s | val_acc %s " % (epoch, train_acc, validation_acc))
        
        if validation_acc > best_accuracy:
            # A network with the best validation accuracy is returned.  
            net_best = net
            net_with_softmax_best = net_with_softmax
            best_accuracy = validation_acc
        
    return net_best, net_with_softmax_best

##################################
###
### Training
###
##################################

def train(channel_input_dirs, hyperparameters, hosts, num_cpus, num_gpus, output_data_dir, model_dir, **kwargs):
    print(sys.version)
    print(sys.executable)
    print(sys.version_info)
    print(mx.__version__)

    '''
    # Load preprocessed data #
    Due to the memory limit of m4 instance, we only use a part of dataset to train the model. 
    '''
    trn_im, test_im, trn_output, test_output = load_data(os.path.join(channel_input_dirs['dataset']))
    
    '''
    # Additional Data Augmentation #
    '''
    # Mirror
    trn_im_mirror = trn_im[:,:,:,::-1]
    trn_output_mirror = np.zeros(trn_output.shape)
    trn_output_mirror[:,0] = trn_output[:,0] 
    trn_output_mirror[:,1] = trn_output[:,1] * -1
    trn_im = np.concatenate((trn_im, trn_im_mirror), axis = 0) 
    trn_output = np.concatenate((trn_output, trn_output_mirror), axis = 0) 
    
    # Color Shift
    for i0 in range(trn_im.shape[0]):
        im_temp = trn_im[i0,:,:,:]
        im_temp = np.transpose(im_temp, (1,2,0)) * 255 #transposing and restoring the color
        im_temp = shiftHSV(im_temp,
                           h_shift_lim=(-0.1, 0.1),
                           s_shift_lim=(-0.1, 0.1),
                           v_shift_lim=(-0.1, 0.1))
        im_temp = np.transpose(im_temp, (2,0,1)) / 255 #transposing and restoring the color
        trn_im[i0,:,:,:] = im_temp
    
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
    
    '''
    Train the model 
    '''
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    
    batch_size = 64
    train_iter = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset((trn_im.astype(np.float32)-0.5) *2, np_trn_grid_cls),
                                                batch_size=batch_size, shuffle=True, last_batch='discard')
    test_iter = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset((test_im.astype(np.float32)-0.5) *2 , np_test_grid_cls),
                                                batch_size=batch_size, shuffle=True, last_batch='discard')
    # Modify the number of output classes
    
    pretrained_net = resnet50_v1(pretrained=True, prefix = 'headpose_')
    net = resnet50_v1(classes=9, prefix='headpose_') 
    net.collect_params().initialize()
    net.features = pretrained_net.features
    
    #net.output.initialize(init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)) # MXNet 1.1.0
    net.initialize(init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)) # MXNet 0.12.1
    
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': float(hyperparameters.get("learning_rate", 0.0005))})
    
    # Fine-tune the model
    logging.getLogger().setLevel(logging.DEBUG)

    num_epoch = 5
    
    print('training started')
    net, net_with_softmax = train_util(output_data_dir, net, train_iter, test_iter, loss, trainer, ctx, num_epoch, batch_size)
    print('training is done')

    ### Serial format v. Modular format 
    # "net" is a serial (i.e. Gluon) network. 
    # In order to save the network in the modular format, "net" needs to be passed to save function. 
    
    return net


def save(net, model_dir):
    '''
    Save the model in the modular format. 
    
    :net: serialized model returned from train
    :model_dir: model_dir The directory where model files are stored.
    
    DeepLens requires a model artifact in the modularized format (.json and .params)
    
    '''
    
    ### Save modularized model 
    # Export .json and .params files
    # model-symbol.json does not come with softmax layer at the end. 
    net.export('{}/model'.format(model_dir)) 
    # Overwrite model-symbol.json with the one with softmax
    net_with_softmax = net(mx.sym.var('data'))
    net_with_softmax = mx.sym.SoftmaxOutput(data=net_with_softmax, name="softmax")
    net_with_softmax.save('{}/model-symbol.json'.format(model_dir)) 



# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #


def model_fn(model_dir):
    """
    Load the model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model 
    """
    
    model_symbol = '{}/model-symbol.json'.format(model_dir)
    model_params = '{}/model-0000.params'.format(model_dir)

    sym, arg_params, aux_params = load_model(model_symbol, model_params)
    ### DSHAPR = (1,3,84,84)
    # The shape of input image. 
    dshape = [('data', DSHAPE)]

    ctx = mx.cpu() # USE CPU to predict... 
    net = mx.mod.Module(symbol=sym,context=ctx)
    net.bind(for_training=False, data_shapes=dshape)
    net.set_params(arg_params, aux_params)
    
    return net 
    
