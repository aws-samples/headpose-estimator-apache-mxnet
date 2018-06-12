## Headpose Estimator Apache Mxnet

Head Pose estimator using Apache MXNet. HeadPose_ResNet50_Tutorial.ipynb helps you to walk through an end-to-end work flow of developing a CNN model from the scratch including data augmentation, fine-tuning, saving check-point model artifacts, validation and inference.

## Preprocessing head-pose data

Please run the following command first to prepare the input data file. 

> python2 preprocessingDataset_py2.py --num-data-aug 15 --aspect-ratio 1

## HeadPose_ResNet50_Tutorial

Jupyter notebook to develop Headpose Estimator CNN model using Apache MXNet. 

## HeadPose_ResNet50_Tutorial_Gluon

Jupyter notebook to develop Headpose Estimator CNN model using Gluon. 

## HeadPose_SageMaker_PythonSDK

Two sets of SageMaker notebooks and entry point scripts to develop the Headpose Estimator model on Amazon SageMaker. 

* **HeadPose_SageMaker_PySDK.ipynb:** SageMaker notebook to invoke an entry point python script. 

* **EntryPt-headpose.py:** An entry point python script to train Headpose Estimator model. This entry point script is analogous to HeadPose_ResNet50_Tutorial.ipynb.
 
* **EntryPt-headpose-wo-cv2.py:** The entry point script without cv2. 

* **HeadPose_SageMaker_PySDK-Gluon.ipynb:** SageMaker notebook to invoke an entry point python script. 

* **EntryPt-headpose-Gluon.py:** An entry point python script to train Headpose Estimator model. This entry point script is analogous to HeadPose_ResNet50_Tutorial_Gluon.ipynb.
 
* **EntryPt-headpose-Gluon-wo-cv2.py:** The entry point script without cv2. 

* **tensorflow_resnet_headpose_for_deeplens.ipynb:** SageMaker notebook to invoke an TensorFlow entry point python script. 

* **resnet_headpose.py:** The TensorFlow main entry point script used for training and hosting

* **resnet_model_headpose.py:** TensorFlow ResNet model

## testIMs

Sample head images for inference test. 

## License

This library is licensed under the Apache 2.0 License. 
