# Image Similarity Model  based on  Image Classification
we propose, a Image Similarity method based on Image Classification. that It ensures semantic consistency between the original and similarity images, and achieves good image diversity. 
This model is trained using Image Classifier  and Resnet18 in such a way that it can retrieve the similar images of the query image.

**Steps :**
1. Image Classifier

    cifar10 Training an image classifier:
    For this tutorial, we will use the CIFAR10 dataset. It has the classes: ‘airplane’,‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
    The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels  in size.

     We will do the following steps in order:
     *	Load and normalizing the CIFAR10 training and test datasets using torchvision.
     *	Define a Convolution Neural Network   
     *	Define a loss function 
     *	Train the network on the training data
     *	Test the network on the test data
     *	Loading and normalizing CIFAR10

2. Resnet18

   First, we predict the class of the input images with using a image classifier model that pre-trained. then each image is placed in a folder with its class images, this folders is given as input to the Resnet18 
   network one by one.Based on the resnet18 implementation from PyTorch it creates feature vectors from the input images, compares it to the other images and sorts for each image a similarity list. Eventually, the 
   result is visualized for a tiny test set that is provided within the repository.





   

**Architecture of Image Similarity method based on Image Classification:**



 ![Image Similarity with classification](https://github.com/Mahmoudi1993/Image-Similarity-Model-based-on-Image-Classification/assets/74957886/2915836e-ac62-4611-b9d4-73762c857000)

# Getting Started   
1. Clone this repo:
```
git clone https://github.com/Mahmoudi1993/Image-Similarity-Model-based-on-Image-Classification.git
cd Image-Similarity-Model-based-on-Image-Classification
```
2. Install conda.
3. Install all the dependencies
```
conda env create --file env.yml
```
4. Switch to the conda environment
```
conda activate Image-Similarity-Model-based-on-Image-Classification
```
5. Install other dependencies
```
sh scripts/install.sh
```
# Pretrained Model
the pretrained models classificatin would place in checkpoints.

# Command Line Args & Reference
**Command Line Args**
```
Measure_Similar_Images.py:

  --Dataset: path to input dataset
    (default: None)
  --data: path to test data(CIFAR-10 dataset)
    (default: None)
  --model: path to weights model
    (default: './checkpoint/cifar_net.pth')
  --inputDir: path to folders are given as input to the resnet18 network
    (default: ./inputImagesCNN)
  --epoch: An epoch means training the neural network with all the training data for one cycle
    (default: 0)
 --Pickle: test_batch
    (default: None)

```

**Reference**

  * https://medium.com/@nagam808surya/cifar-10-image-classification-using-pytorch-90332041ba1f
  * https://towardsdatascience.com/effortlessly-recommending-similar-images-b65aff6aabfb
  * https://towardsdatascience.com/recommending-similar-images-using-pytorch-da019282770c

