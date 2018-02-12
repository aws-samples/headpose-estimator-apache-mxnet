## Headpose Estimator Apache Mxnet

Head Pose estimator using Apache MXNet. HeadPose_ResNet50_Tutorial.ipynb helps you to walk through an end-to-end work flow of developing a CNN model from the scratch including data augmentation, fine-tuning, saving check-point model artifacts, validation and inference.

## Preprocessing head-pose data

Please run the following command first to prepare the input data file. 

> python2 preprocessingDataset_py2.py --num-data-aug 15 --aspect-ratio 1

## HeadPose_ResNet50_Tutorial

Jupyter notebook to develop Headpose Estimator CNN model using Apache MXNet. 

## HeadPose_SageMaker_PythonSDK

A set of SageMaker notebook and entry point script to develop the Headpose Estimator model on Amazon SageMaker. 

* **HeadPose_SageMaker_PySDK.ipynb:** SageMaker notebook to invoke an entry point python script. 

* **EntryPt-headpose.py:** An entry point python script to train Headpose Estimator model. This entry point script is analogous to HeadPose_ResNet50_Tutorial.ipynb.
 
* **EntryPt-headpose-wo-cv2.py:** The entry point script without cv2. 

## testIMs

Sample head images for inference test. 

## License

This library is licensed under the Apache 2.0 License. 
