## Headpose Estimator Apache Mxnet

Head Pose estimator using Apache MXNet. HeadPose_ResNet50_Tutorial.ipynb helps you to walk through an end-to-end work flow of developing a CNN model from the scratch including data augmentation, fine-tuning, saving check-point model artifacts, validation and inference.

## Preprocessing head-pose data

Please run the following command first to prepare the input data file. 

> python2 preprocessingDataset_py2.py --num-data-aug 15 --aspect-ratio 1

## HeadPose_ResNet50_Tutorial

Jupyter notebook to develop Headpose Estimator CNN model using Apache MXNet. 

## HeadPose_SageMaker_PythonSDK

A set of SageMaker notebook and entry point script to repeat the development of Headpose Estimator model. 

* HeadPose_SageMaker_PySDK.ipynb 
* EntryPt-headpose.py 
* EntryPt-headpose-wo-cv2.py

## testIMs

Sample head images for inference test. 

## License

This library is licensed under the Apache 2.0 License. 
