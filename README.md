# Waymo 2D Object Detection Optimization
[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6%20%7C%203.7-blue)](https://www.python.org/downloads/release/python-360/)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)

Enhancing 2D object detection by optimizing anchor generation and addressing class imbalance. 

This project uses the TensorFlow Object Detection API and proposes several modifications to the standard Faster R-CNN 
implementation, improving 2D detection over the Waymo Open Dataset:

 * Per-region anchor optimization using genetic algorithms
 * Spatial ROI features in the second-stage Fast R-CNN header network
 * Reduced focal loss to improve performance over minority and difficult instances
 * Ensemble models using non-maximum suppression
<br/><br/>  

<p align="center">
  <img width="800" height="300" src="https://github.com/carranza96/waymo-detection-optimization/blob/master/doc/images/FasterRCNN.png">
</p>

## Installation

This project depends on several libraries and contains two submodules that have to be installed:
  * [TensorFlow Object Detection API](https://github.com/carranza96/models/): The *models* folder contains a forked repository with the proposed modifications
  * [Waymo repository](https://github.com/carranza96/waymo-open-dataset/) for reading the dataset and computing metrics
  
Full details on the installation steps and system requirements can be found at [installation.md](https://github.com/carranza96/waymo-detection-optimization/blob/master/doc/installation.md)


## Getting started
### Scripts
The *scripts* folder provides ready-to-use shell scripts for many operations:
 * [Convert Waymo dataset to the format required by the TF Object Detection API](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/convert_waymo_tfrecord.sh)
 * [Training models](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/od_api/train_script.sh)
 * [Exporting inference graphs](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/od_api/export_script.sh)
 * [Infer predictions](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/od_api/inference_script.sh)
 * [Calculating inference time](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/average_inference_time.sh)
 * [Ensemble predictions](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/ensemble_predictions.sh)
 * [Computing Waymo metrics](https://github.com/carranza96/waymo-detection-optimization/blob/master/src/scripts/waymo_evaluation/detection_metrics.sh)

### Example model configuration
An example Faster R-CNN model configuration is provided in the file [pipeline.config](https://github.com/carranza96/waymo-detection-optimization/blob/master/saved_models/optimized_faster_rcnn/pipeline.config), using the proposed improvements: anchor optimization, spatial ROI features, and reduced focal loss
## Authors <a name="authors"></a>

* **Pedro Lara-Benítez** - [LinkedIn](www.linkedin.com/in/pedrolarben)
* **Manuel Carranza-García** - [LinkedIn](https://www.linkedin.com/in/manuelcarranzagarcia96/)
* **Jorge García-Gutiérrez** 
* **José C. Riquelme**

## License<a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details




