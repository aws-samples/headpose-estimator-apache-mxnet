

'''
Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

'''

# # Data Parsing
# 
# This notebook is to pre-process Prima head-pose dataset. The output ``pickle`` file will be used for ``HeadPose_ResNet50_Tutorial``.
# 
# Original Data: http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html
# 
# > N. Gourier, D. Hall, J. L. Crowley,
# > Estimating Face Orientation from Robust Detection of Salient Facial Features,
# > *Proceedings of Pointing 2004, ICPR, International Workshop on Visual Observation of Deictic Gestures*, Cambridge, UK

import argparse
import os
import sys
import numpy as np
import urllib
import cv2

import pickle
import tarfile
from glob import glob

parser = argparse.ArgumentParser(description='Head Pose Preprocessing')

parser.add_argument('--num-data-aug', type=int, default=15,
                    help='number of augmentation on train data (default: 15)')
parser.add_argument('--num-data-aug-val', type=int, default=3,
                    help='number of augmentation on validation data (default: 3)')
parser.add_argument('--aspect-ratio', type=int, default=1,
                    help='aspect_ratio of output image. 1: 84 pix x 84 pix, 0: 96 pix x 54 pix (default: 1)')


opt = parser.parse_args()

num_data_aug = opt.num_data_aug
print("Number of data augmentation: ", num_data_aug)
num_data_aug_val = opt.num_data_aug_val
'''
aspect_ratio = 0 -> 16:9 -> 96x54
aspect_ratio = 1 -> 1:1 -> 84x84
'''
aspect_ratio = opt.aspect_ratio
if aspect_ratio == 1:
    print("Aspect Ratio 1:1 (84 pix x 84 pix) ")
else:
    print("Aspect Ratio 16:9 (96 pix x 54 pix) ")

# ## Download dataset
print('**----  Downloading Dataset  ----**')

def download_data(url, force_download=True): 
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

url_ds1 = "http://www-prima.inrialpes.fr/perso/Gourier/Faces/HeadPoseImageDatabase.tar.gz"
fname = download_data(url_ds1) ### 28MB
print(fname)

# HeadPoseImageDatabase.tar.gz

with tarfile.open(fname, "r:gz") as tar:
    tar.extractall()


### http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html
lst_persons = [str(i + 1).zfill(2) for i in range(15)]
lst_ser = [str(i + 1).zfill(1) for i in range(2)]
i_num = 92
lst_tilt = ['-90','-60','-30','-15','0','+15','+30','+60','+90'] ### 9 elements
lst_pan = ['-90','-75','-60','-45','-30','-15','0','+15','+30','+45','+60','+75','+90'] ### 13 elements

home_dir = os.getcwd()


# 3 out of 15 subjects are chosen for the validation data.

sub_test = [4,9,14]


# ## Training Data
print('**----  Training Data  ----**')

data_filename = []
data_tilt = []
data_pan = []
data_persons = []
data_txtfile = []
i_count = 0
for i0 in range(len(lst_persons)):
    pre_path = home_dir + "/Person" + lst_persons[i0]
    # Change Dir
    os.chdir(pre_path)
    if i0 not in sub_test:
        print(i0)
        for i1 in range(len(lst_ser)):
            for i2 in range(i_num + 1):
                pre_file = 'person' + lst_persons[i0] + lst_ser[i1] + str(i2).zfill(2)
                matches = [f for f in os.listdir(pre_path) if f.startswith(pre_file)]

                if len(matches[0][11:-4].split('0')[0]) <= 2: 
                    ## The tilt is either -90,-60,-30,0,+30,+60,or+90.
                    s_tilt = matches[0][11:-4].split('0')[0] + '0'
                    s_pan = matches[0][:-4].split(matches[0][:11] + s_tilt)[1]
                elif matches[0][11:-4].split('0')[0][:3] == '+15':
                    ## The tilt is +15
                    s_tilt = '+15'
                    s_pan = matches[0][:-4].split(matches[0][:11] + s_tilt)[1]
                elif matches[0][11:-4].split('0')[0][:3] == '-15':
                    ## The tilt is +15
                    s_tilt = '-15'
                    s_pan = matches[0][:-4].split(matches[0][:11] + s_tilt)[1]
                data_filename = data_filename + [matches[0][:-4]]
                data_tilt = data_tilt + [int(s_tilt)]
                data_pan = data_pan + [int(s_pan)]
                data_persons = data_persons + [int(i0)]
                
                ### Text file contains the face label. 
                txtfile = open(matches[0][:-4] + ".txt", 'r') 
                data_txtfile = data_txtfile + [txtfile.read().splitlines()[3:]]

                im = cv2.imread(matches[0][:-4] + ".jpg")

                im = im.reshape(im.shape[0],im.shape[1],im.shape[2],1).astype(np.float32)/255 # Normalized
                if i_count == 0:
                    data_im_concat = im
                else:
                    data_im_concat = np.concatenate((data_im_concat, im), axis = 3)    
                i_count += 1

### Data Augmentation 
n_aug = num_data_aug 
i_count = 0

for i0 in range(n_aug):
    print(i0)

    for i1 in range(data_im_concat.shape[3]):
        ### Cropping
        centx = int(data_txtfile[i1][0])
        centy = int(data_txtfile[i1][1])
        x_move = int(data_txtfile[i1][2]) // 2
        y_move = int(data_txtfile[i1][3]) // 2
        ## ImageCrop 
        ## crop_ulX should be somewhere between (0,0) and top-left courner of the face 
        ## i.e. (centx - x_move, centy - y_move)
        while True: 
            crop_ulx = int(np.random.random_sample()*(centx- x_move))
            crop_uly = int(np.random.random_sample()*(centy- y_move))
        
            min_height = centy + y_move - crop_uly
            max_height = data_im_concat.shape[0] - crop_uly
            if aspect_ratio == 0:
                # Apect Ratio 16:9
                crop_height = (min_height + int(np.random.random_sample()*(max_height - min_height)))//9 * 9
                crop_width = crop_height // 9 * 16
            else:
                # Apect Ratio 1:1
                crop_height = (min_height + int(np.random.random_sample()*(max_height - min_height)))//9 * 9
                crop_width = crop_height // 9 * 9                    
        
            if crop_ulx + crop_width > centx + x_move and crop_ulx + crop_width < data_im_concat.shape[1]             and crop_uly + crop_height > centy + y_move and crop_uly + crop_height < data_im_concat.shape[0]:
                break
        im = data_im_concat[:,:,:,i1]
        im_crop = im[crop_uly:crop_uly + crop_height, crop_ulx:crop_ulx + crop_width]
        if aspect_ratio == 0:
            # Apect Ratio 16:9
            im_crop = cv2.resize(im_crop, (96, 54))
        else:
            # Apect Ratio 1:1
            im_crop = cv2.resize(im_crop, (84, 84))
        im_crop = im_crop.reshape(im_crop.shape[0],im_crop.shape[1],im_crop.shape[2],1).astype(np.float32)
        if i_count == 0:
            data_im_concat_aug = im_crop
        else:
            data_im_concat_aug = np.concatenate((data_im_concat_aug, im_crop), axis = 3)
        i_count += 1



print(data_im_concat_aug.shape)


### Concatinating the output
np_data_tilt_temp = np.asarray(data_tilt).reshape(len(data_tilt),1).astype(np.float32)/90 # Normalized
np_data_pan_temp = np.asarray(data_pan).reshape(len(data_pan),1).astype(np.float32)/90 # Normalilzed

for i0 in range(n_aug):
    if i0 == 0:
        np_data_tilt = np_data_tilt_temp
        np_data_pan = np_data_pan_temp
    else:
        np_data_tilt = np.concatenate((np_data_tilt, np_data_tilt_temp), axis = 0)
        np_data_pan = np.concatenate((np_data_pan, np_data_pan_temp), axis = 0)

data_output = np.concatenate((np_data_tilt, np_data_pan), axis = 1)
print(data_output.shape)


### Transpose the data
## MXNET input 4D (batch_size, num_channels, height, width) ==> (bsize, 3, height, width)
data_im_concat_aug_t = np.transpose(data_im_concat_aug, (3,2,0,1))
data_im_concat_aug_t.shape


trn_im = data_im_concat_aug_t
trn_output = data_output


# ## Validation Data
print('**----  Validation Data  ----**')
data_filename = []
data_tilt = []
data_pan = []
data_persons = []
data_txtfile = []
i_count = 0
for i0 in range(len(lst_persons)):
    pre_path = home_dir + "/Person" + lst_persons[i0]
    # Change Dir
    os.chdir(pre_path)
    if i0 in sub_test:
        print(i0)
        for i1 in range(len(lst_ser)):
            for i2 in range(i_num + 1):
                pre_file = 'person' + lst_persons[i0] + lst_ser[i1] + str(i2).zfill(2)
                matches = [f for f in os.listdir(pre_path) if f.startswith(pre_file)]

                if len(matches[0][11:-4].split('0')[0]) <= 2: 
                    ## The tilt is either -90,-60,-30,0,+30,+60,or+90.
                    s_tilt = matches[0][11:-4].split('0')[0] + '0'
                    s_pan = matches[0][:-4].split(matches[0][:11] + s_tilt)[1]
                elif matches[0][11:-4].split('0')[0][:3] == '+15':
                    ## The tilt is +15
                    s_tilt = '+15'
                    s_pan = matches[0][:-4].split(matches[0][:11] + s_tilt)[1]
                elif matches[0][11:-4].split('0')[0][:3] == '-15':
                    ## The tilt is +15
                    s_tilt = '-15'
                    s_pan = matches[0][:-4].split(matches[0][:11] + s_tilt)[1]
                data_filename = data_filename + [matches[0][:-4]]
                data_tilt = data_tilt + [int(s_tilt)]
                data_pan = data_pan + [int(s_pan)]
                data_persons = data_persons + [int(i0)]
                
                ### Text file contains the face label. 
                txtfile = open(matches[0][:-4] + ".txt", 'r') 
                data_txtfile = data_txtfile + [txtfile.read().splitlines()[3:]]

                im = cv2.imread(matches[0][:-4] + ".jpg")

                im = im.reshape(im.shape[0],im.shape[1],im.shape[2],1).astype(np.float32)/255 # Normalized
                if i_count == 0:
                    data_im_concat = im
                else:
                    data_im_concat = np.concatenate((data_im_concat, im), axis = 3)    
                i_count += 1

### Data Augmentation 
n_aug = num_data_aug_val
i_count = 0

for i0 in range(n_aug):
    print(i0)

    for i1 in range(data_im_concat.shape[3]):
        ### Cropping
        centx = int(data_txtfile[i1][0])
        centy = int(data_txtfile[i1][1])
        x_move = int(data_txtfile[i1][2]) // 2
        y_move = int(data_txtfile[i1][3]) // 2
        ## ImageCrop 
        ## crop_ulX should be somewhere between (0,0) and top-left courner of the face 
        ## i.e. (centx - x_move, centy - y_move)
        while True: 
            crop_ulx = int(np.random.random_sample()*(centx- x_move))
            crop_uly = int(np.random.random_sample()*(centy- y_move))
        
            min_height = centy + y_move - crop_uly
            max_height = data_im_concat.shape[0] - crop_uly
            if aspect_ratio == 0:
                # Apect Ratio 16:9
                crop_height = (min_height + int(np.random.random_sample()*(max_height - min_height)))//9 * 9
                crop_width = crop_height // 9 * 16
            else:
                # Apect Ratio 1:1
                crop_height = (min_height + int(np.random.random_sample()*(max_height - min_height)))//9 * 9
                crop_width = crop_height // 9 * 9                    
        
            if crop_ulx + crop_width > centx + x_move and crop_ulx + crop_width < data_im_concat.shape[1]             and crop_uly + crop_height > centy + y_move and crop_uly + crop_height < data_im_concat.shape[0]:
                break
        im = data_im_concat[:,:,:,i1]
        im_crop = im[crop_uly:crop_uly + crop_height, crop_ulx:crop_ulx + crop_width]
        if aspect_ratio == 0:
            # Apect Ratio 16:9
            im_crop = cv2.resize(im_crop, (96, 54))
        else:
            # Apect Ratio 1:1
            im_crop = cv2.resize(im_crop, (84, 84))
        im_crop = im_crop.reshape(im_crop.shape[0],im_crop.shape[1],im_crop.shape[2],1).astype(np.float32)
        if i_count == 0:
            data_im_concat_aug = im_crop
        else:
            data_im_concat_aug = np.concatenate((data_im_concat_aug, im_crop), axis = 3)
        i_count += 1

print(data_im_concat_aug.shape)


### Concatinating the output
np_data_tilt_temp = np.asarray(data_tilt).reshape(len(data_tilt),1).astype(np.float32)/90 # Normalized
np_data_pan_temp = np.asarray(data_pan).reshape(len(data_pan),1).astype(np.float32)/90 # Normalilzed

for i0 in range(n_aug):
    if i0 == 0:
        np_data_tilt = np_data_tilt_temp
        np_data_pan = np_data_pan_temp
    else:
        np_data_tilt = np.concatenate((np_data_tilt, np_data_tilt_temp), axis = 0)
        np_data_pan = np.concatenate((np_data_pan, np_data_pan_temp), axis = 0)

data_output = np.concatenate((np_data_tilt, np_data_pan), axis = 1)
print(data_output.shape)

### Transpose the data
## MXNET input 4D (batch_size, num_channels, height, width) ==> (bsize, 3, height, width)
data_im_concat_aug_t = np.transpose(data_im_concat_aug, (3,2,0,1))
data_im_concat_aug_t.shape

test_im = data_im_concat_aug_t
test_output = data_output


print(test_im.shape, trn_im.shape)
print(test_output.shape, trn_output.shape)

pickle_name = "HeadPoseData_trn_test_x{}_py2.pkl".format(num_data_aug)

import pickle
os.chdir(home_dir)
with open(pickle_name, "wb") as f:
    pickle.dump((trn_im, test_im, trn_output, test_output), f)
    
print(pickle_name, ' is saved!')
print('**----  Done  ----**')


# # End 
