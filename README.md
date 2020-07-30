# Waymo 2D Object Detection Optimization
[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
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
  <img width="1000" height="368" src="https://github.com/carranza96/waymo-detection-optimization/blob/master/doc/images/FasterRCNN.png">
</p>

## Installation


## Authors <a name="authors"></a>

* **Pedro Lara-Benítez** - [LinkedIn](www.linkedin.com/in/pedrolarben)
* **Manuel Carranza-García** - [LinkedIn](https://www.linkedin.com/in/manuelcarranzagarcia96/)
* **Jorge García-Gutiérrez** 
* **José C. Riquelme**

## License<a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details




