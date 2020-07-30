### System requirements
 * Python 3.6 | 3.7
 * Protobuf compiler >= 3.0
 * g++ 5 or higher.
 * NVIDIA® GPU drivers — CUDA 10.0 requires 410.x or higher.
 * CUDA® Toolkit 10.0 
 * cuDNN SDK (>= 7.4.1)
 
### Python libraries
```bash
pip install docs/requirements.txt
```
#### TF Object Detection API
Install the modified fork provided in the models folder by executing the following commands:
```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install .
```

```bash
# Test the installation.
python object_detection/builders/model_builder_tf1_test.py
```

#### Waymo library
```bash
pip3 install waymo-open-dataset-tf-1-15-0==1.2.0
```

```bash
# Configure Bazel environment
cd waymo-open-dataset
BAZEL_VERSION=0.28.0
wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
./configure.sh
```