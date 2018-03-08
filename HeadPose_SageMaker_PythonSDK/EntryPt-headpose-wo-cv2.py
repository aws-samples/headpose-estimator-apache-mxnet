import logging

import mxnet as mx
import numpy as np
import os
import urllib
import pickle
import sys
#import cv2

##################################
###
### Helper functions
###
##################################

class EvalCallback(object):
    '''
    Attempt at a Earlystopping solution using the first metric.
    
    pass an instance of the metric or the instance name to specify which metric to use for stopping
    
    1. epoch_end_callback: doesn't provide the metrics to the registered callback function, hence we can't use it to track
    metrics and save
    
    2. eval_end_callback: while it provides us with eval metrics, there isn't a clean way to stop the training, so the best
    thing to do is track and save the best model we have seen so far based on the metric that operator defined.
    
    '''
    def __init__(self, model_prefix, metric, op="max", save_model=True, patience=0, delta=0):
        assert isinstance(metric, str) or isinstance(metric,mx.metric.EvalMetric), "Metric must be the name or the instance"
        self.metric_name = metric if isinstance(metric,str) else metric.name
        self.model_prefix = model_prefix
        self.eval_metrics = []
        self.save_model = save_model
        self.metric_op = np.less if op == "min" else np.greater
        self.best_metric = np.Inf if self.metric_op == np.less else -np.Inf
        self.delta = delta #min difference between metric changes
        self.patience = patience

    def get_loss_metrics(self):
        return self.eval_metrics
    
    def __call__(self, param):
        cur_epoch = param.epoch
        module_obj = param.locals['self']
        name_values = param.eval_metric.get_name_value()
        
        names, cur_values = zip(*name_values)
        if self.metric_name not in names:
            print("Metric %s not in model metrics: %s" % (self.metric_name, names))
            return
        name, cur_value = name_values[names.index(self.metric_name)]
        self.eval_metrics.append(cur_value)
        if cur_epoch >= self.patience:
        #print cur_value, self.best_metric, self.metric_op(cur_value - self.delta, self.best_metric)
            if self.metric_op(cur_value - self.delta, self.best_metric):
                self.best_metric = cur_value
                print('The best model found so far at epoch %05d with %s %s' % (cur_epoch, name, cur_value))
                if self.save_model:
                    logging.info('Saving the Model')    
                    module_obj.save_checkpoint(self.model_prefix, cur_epoch)
                    param_fname = '%s-%04d.params' % (self.model_prefix, cur_epoch)
                    os.rename(param_fname, '%s-0000.params' % self.model_prefix ) #rename the model

#cv2 is not supported in ml.m4 instance
#cv2 is not supported in inference instance
#def shiftHSV(im, h_shift_lim=(-180, 180),
#                 s_shift_lim=(-255, 255),
#                 v_shift_lim=(-255, 255), drop_p=0.5):
#    if np.random.random() < drop_p:
#        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#        h, s, v = cv2.split(im)
#        h_shift = np.random.uniform(h_shift_lim[0], h_shift_lim[1])
#        h = cv2.add(h, h_shift) 
#        s_shift = np.random.uniform(s_shift_lim[0], s_shift_lim[1])
#        s = cv2.add(s, s_shift)
#        v_shift = np.random.uniform(v_shift_lim[0], v_shift_lim[1])
#        v = cv2.add(v, v_shift)
#        im = cv2.merge((h, s, v))
#        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
#        im = np.uint8(im) 
#        im = np.float32(im)
#    return im                     
                                       
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

def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))
        
def get_graph():
    ## ResNet-50 from Model Zoo
    get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)
    return sym, arg_params, aux_params


def change_num_output(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = {k:arg_params[k] for k in arg_params if 'fc1' not in k}
    return (net, new_args)
    

def angles2Cat(angles_thrshld, angl_input):
    # angl_input: Normalized angle -90 - 90 -> -1.0 - 1.0
    angles_cat_temp = angles_thrshld + [angl_input]
    return np.argmin(np.multiply(sorted(angles_cat_temp)-angl_input,sorted(angles_cat_temp)-angl_input))

##################################
###
### Training
###
##################################

def train(channel_input_dirs, hyperparameters, hosts, num_cpus, num_gpus, output_data_dir, **kwargs):
    print(sys.version)
    print(sys.executable)
    print(sys.version_info)

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
    #for i0 in range(trn_im.shape[0]):
    #    im_temp = trn_im[i0,:,:,:]
    #    im_temp = np.transpose(im_temp, (1,2,0)) * 255 #transposing and restoring the color
    #    im_temp = shiftHSV(im_temp,
    #                       h_shift_lim=(-0.1, 0.1),
    #                       s_shift_lim=(-0.1, 0.1),
    #                       v_shift_lim=(-0.1, 0.1))
    #    im_temp = np.transpose(im_temp, (2,0,1)) / 255 #transposing and restoring the color
    #    trn_im[i0,:,:,:] = im_temp
    
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
    batch_size = 100
    trn_iter_grid = mx.io.NDArrayIter((trn_im.astype(np.float32)) * 255, np_trn_grid_cls, batch_size, shuffle=True)
    test_iter_grid = mx.io.NDArrayIter((test_im.astype(np.float32)) * 255, np_test_grid_cls, batch_size)
            
    # Modify the number of output classes
    sym, arg_params, aux_params = get_graph()
    (new_sym, new_args) = change_num_output(sym, arg_params, num_classes = n_grid * n_grid)
    
    # Fine-tune the model
    logging.getLogger().setLevel(logging.DEBUG)
    kvstore = 'local' if len(hosts) == 1 else 'dist_sync'
    num_epoch = 5
    
    net = mx.mod.Module(symbol=new_sym, context=get_train_context(num_cpus, num_gpus))
    net.bind(data_shapes=[trn_iter_grid.provide_data[0]], label_shapes=[trn_iter_grid.provide_label[0]])
    
    ### Checkpoint
    # output_data_dir -> Files will be saved in output.tar.gz
    model_prefix = output_data_dir +'chkpt_Res50_1_3x3'
    checkpoint = mx.callback.do_checkpoint(model_prefix, 10)
 
    ### EvalCallback
    # output_data_dir -> Files will be saved in output.tar.gz
    model_prefix_best = output_data_dir + 'chkpt_best_Res50_1_3x3'
    ev_cb = EvalCallback(model_prefix_best, "accuracy", "max", save_model=True)
    
    net.fit(trn_iter_grid,
                  eval_data=test_iter_grid,
                  arg_params=new_args, ### Fine Tune 
                  aux_params=aux_params, ### Fine Tune
                  initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ### Fine Tune
                  allow_missing=True, ### Fine Tune 
                  kvstore=kvstore,
                  optimizer='adam',
                  optimizer_params={'learning_rate': float(hyperparameters.get("learning_rate", 0.0005))},
                  eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  epoch_end_callback=checkpoint,
                  eval_batch_end_callback=ev_cb, ##
                  num_epoch=num_epoch)
    return net


def get_train_context(num_cpus, num_gpus):
    if num_gpus > 0:
        return [mx.gpu(i) for i in range(num_gpus)]
    return mx.cpu()
