# tf-onnx-sample

Welcome reader!

This is a small project to learn more about Tensorflow and ONNX.
There are two main projects here, inside the **model** folder there is Tensorflow model created to classify images, and inside the **inference** folder a C# application to use the model.

Inside the **onnx** you will find a notebook to learn more about ONNX.

Feel free to use this content. Happy learning!

## model

This is where everything starts. Here you will find a Python program to define, train, and export a Tensorflow model.
The model will be exported as ONNX.

The model will classify images from different classes. In order to train the model you must create a folder like the dataset here in this repo, where each subdirectory is a class. Inside this subdirectory add the images of that class.

Then execute the following command from this repo root folder. It will train the model and output it inside the **output** folder.

```
python model/main.py 
    --dataset-dir=<dir> 
    --image-width=<width> 
    --image-height=<height> 
    --image-channels=<channels> 
    --batch-size=<batch> 
    --epochs=<epochs>
```

The parameters are:
- **--dataset-dir**: the directory where the images organized by class are located.
- **--image-width**: image width to be used.
- **--image-height**: image height to be used.
- **--image-channels**: image channels to be used.
- **--batch-size**: batch size to be used.
- **--epochs**: epochs to train the model.

#### Notes

- The model will resize all images to the specified size. The width and height define the size. The larger the image, the bigger the model will be, the more time will take to train, and the more memory it will need.

- The **batch-size** parameter defines when we should calculate the loss and tune the model. I usually use 32.

- The **epochs** parameter defines how many rounds we should go over all the images to train the model.



## inference

This is a C# application used to classify images. It uses the model created from the Python program described earlier or any other ONNX model that has the same input and output signature.
The application uses the C# ONNX runtime to do its "magic".

```
inference.exe 
    --gpu
    --verbose
    --model <onnx model path>
    --image <image to classify>
    --classes <classes file to return a human-readable class name>
```

The parameters are:

- **--gpu**: enable the use of GPU. All DLLs required must be present in the PATH. Take a look at the section **ONNX Runtime - GPU Execution Provider**.
- **--verbose**: enable ONNX verbose logging.
- **--model**: ONNX model to be used.
- **--image**: image to be classified.
- **--classes**: classes file to be used.

## onnx

ONNX is a standard created by several companies to share easily models between different frameworks and runtimes. Please take a look [here](./onnx//onnx.ipynb) for more detailed information.

## ONNX Runtime - GPU Execution Provider

To use NVIDIA GPU during the inference for C# applications we need the **Microsoft.ML.OnnxRuntime.Gpu** package.

This package will require the following packages from NVIDIA. I downloaded them from [here](https://developer.download.nvidia.com/compute/cuda/redist/) and [here](https://developer.nvidia.com/rdp/cudnn-download). I am also providing the links below for the versions I have used.

1. [libcublas-windows-x86_64-11.11.3.6-archive.zip](https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-11.11.3.6-archive.zip)
2. [libcufft-windows-x86_64-10.9.0.58-archive.zip](https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-10.9.0.58-archive.zip)
3. [cudnn-windows-x86_64-8.9.5.29_cuda11-archive](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.5/local_installers/11.x/cudnn-windows-x86_64-8.9.5.29_cuda11-archive.zip/)
4. [cuda_cudart-windows-x86_64-11.8.89-archive.zip](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-11.8.89-archive.zip)

The C# sample in the inference folder already supports the use of GPU. Just enable it with the **--gpu** parameter.

## ONNX Runtime - OpenVINO Execution Provider

To use OpenVINO in ONNX C# runtime we need to install the following packages. They are not available on Nuget.

1. [Microsoft.ML.OnnxRuntime.Managed.1.15.0-dev-20230621-0632-69695172e.nupkg](https://github.com/intel/onnxruntime/releases/download/v5.0.0/Microsoft.ML.OnnxRuntime.Managed.1.15.0-dev-20230621-0632-69695172e.nupkg)
2. [Microsoft.ML.OnnxRuntime.OpenVino.1.15.0-dev-20230621-0632-69695172e.nupkg](https://github.com/intel/onnxruntime/releases/download/v5.0.0/Microsoft.ML.OnnxRuntime.OpenVino.1.15.0-dev-20230621-0632-69695172e.nupkg)

In your C#, code you must enable the OpenVINO EP.

## Neural Network

Below there is an image showing the neural network used to infer the image classification. The image was exported from Netron, and we can see the several nodes, the input, and the output. The trainable nodes will also contain the values learned during the training. You can even export them as numpy arrays from Netron!

The idea of this sample was not to create an innovative neural network, but show end-to-end the process to create, train and infer the model using standard tools and practices.

![model](./assets/model.png)
