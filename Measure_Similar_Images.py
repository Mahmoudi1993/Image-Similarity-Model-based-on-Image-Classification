#  ***guidance for similar images with image classification based on Resnet18***

# This is a basic item-item guidance system implementation. We first predict the class of the input images using a pre-trained image classification model,
# then each image is placed in a folder with its class images, this folder is given as input to the Resnet18 network until the Based on the resnet18
# implementation from PyTorch it creates feature vectors from the input images, compares it to the other images in that folder and sorts for each image from self folder a similarity list. Eventually, the result is visualized for a tiny test set that is provided within the repository.
# ___________________________________________________________________________________

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
import torchvision
import sys, os
from os import listdir
from PIL import Image
import torchvision.transforms as transforms
from six.moves import cPickle as pickle
from absl import app
from absl import flags
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from New_Scratch_image_classification_inference import predictor




FLAGS = flags.FLAGS
flags.DEFINE_string('Dataset', None, './data')
flags.DEFINE_string('data', None, './data/cifar10_raw/images/test/')
flags.DEFINE_string('model', './checkpoint/cifar_net.pth', './checkpoint/cifar_net.pth')
flags.DEFINE_string('inputDir', './inputImagesResnet18', './inputImagesResnet18')
flags.DEFINE_integer('epoch', 0, 'score epoch')
flags.DEFINE_string('pickle', None, './data/data/cifar-10-batches-py/test_batch')



def main(argv):
  path = FLAGS.Dataset
  Path = FLAGS.data
  Model = FLAGS.model
  InputDir = FLAGS.inputDir
  Epoch = FLAGS.epoch
  Pickle = FLAGS.pickle
  
  
  
  # Rescaling
  # We assume to have a folder "data" in the working directory. It shall contain png images. 
  # As we will employ resnet18 using PyTorch, we need to resize the images to normalized 224x224 images 
  # In a first step they are resized and stored in a different folder inputImagesCNN.
  
  
  emp1 = predictor(path, Model, Pickle)
  result = (emp1.get_frame()) 
  Predictor  = str(result)
  
  # Input_Image & Dataset Referance
  import shutil, os
  keys = []
  values = []
  pred_dict =  eval(Predictor)
  items = pred_dict.items()
  for item in items:
      keys.append(item[0]), values.append(item[1])


  #list path
  for index in range(len(keys)):
    InputImagesResnet18 = []

    for folder_name in values:
      inputImagesResnet18 = [] 
      M = Path + folder_name + '/'


      if M[-1] != '/':
          M += '/'

      for file in listdir(M):
          if os.path.isfile(M + file):
              inputImagesResnet18.append(M + file)
          else:
              inputImagesResnet18 += list_images_from_path(M + file)


      InputImagesResnet18.append(inputImagesResnet18)
      #print(len(inputImagesResnet18))
  
  list_files =[]
  for i in range(len(keys)):
    pythonfile = keys[i]
    for root, dirs, files in os.walk(r'./data/cifar10_raw/images/test/'):
      for name in files:
          if name == pythonfile:
              M_1 = str(os.path.abspath(os.path.join(root, name)))
              list_files.append(M_1)

  
  for index in range(len(keys)):
    InputImagesResnet18[index].append(list_files[index])
    
  #save
  keys_dic = keys
  values_dic = InputImagesResnet18
  for i in range(len(keys_dic)):

    inputDir = InputImagesResnet18[i]
    inputDirCNN = 'inputImagesCNN{}'.format(i)

    if not os.path.exists(inputDirCNN):
      os.mkdir(inputDirCNN)

    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    
    for imageName in InputImagesResnet18[i]:

      basename_without_ext = os.path.splitext(os.path.basename(imageName))[0]
      I = Image.open(os.path.join(imageName))

      # needed input dimensions for the CNN
      inputDim = (224,224)
      transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
      newI = transformationForCNNInput(I)

      inputImagesCNN = 'inputImagesCNN' + str(i)
      # copy the rotation information metadata from original image and save, else your transformed images may be rotated
      newI.save(os.path.join(inputImagesCNN, basename_without_ext+"."+"JPEG"))

      newI.close()
      I.close() 
  


   # 1. Creating the similarity matrix with Resnet18
   #    Let us first calculate the feature vectors with resnet18 on a CPU. The input is normalized to the ImageNet mean values/standard deviation.
   # 2. Cosine similarity
   #    Calculate for all vectors the cosine similarity to the other vectors. Note that this matrix may become huge, hence infefficient, with many thousands of images.
   # 3. Prepare top-k lists
   #    Now that the similarity matrix is fully available, the last step is to sort the values per item and store the top similar entries in another data structure.
   # 4. Get and visualize similar images for four example inputs

  import torch
  from tqdm import tqdm
  from torchvision import models
  from torchvision import transforms
  import matplotlib.pyplot as plt
  import pickle

  # 1. **Creating the similarity matrix with Resnet18**
  # for this prototype we use no gpu, cuda= False and as model resnet18 to obtain feature vectors

  class Img2VecResnet18():
      def __init__(self):

          self.device = torch.device("cpu")
          self.numberFeatures = 512
          self.modelName = "resnet-18"
          self.model, self.featureLayer = self.getFeatureLayer()
          self.model = self.model.to(self.device)
          self.model.eval()
          self.toTensor = transforms.ToTensor()

          # normalize the resized images as expected by resnet18
          # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
          self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      def getVec(self, img):
          image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
          embedding = torch.zeros(1, self.numberFeatures, 1, 1)

          def copyData(m, i, o): embedding.copy_(o.data)

          h = self.featureLayer.register_forward_hook(copyData)
          self.model(image)
          h.remove()

          return embedding.numpy()[0, :, 0, 0]

      def getFeatureLayer(self):

          cnnModel = models.resnet18(pretrained=True)
          layer = cnnModel._modules.get('avgpool')
          self.layer_output_size = 512

          return cnnModel, layer

  keys_dic = keys
  values_dic = InputImagesResnet18

  # generate vectors for all the images in the set
  for i in range(len(keys_dic)):
    img2vec = Img2VecResnet18()
    allVectors = {}
    print("Converting images to feature vectors:")
    keys_dic = keys
    values_dic = InputImagesResnet18
    
  #for i in range(len(keys_dic)):
    inputImagesCNN = 'inputImagesCNN' + str(i)
    for image in tqdm(os.listdir(inputImagesCNN)):
        I = Image.open(os.path.join(inputImagesCNN, image))
        vec = img2vec.getVec(I)
        allVectors[image] = vec
        I.close()


    # 2. **Cosine similarity**
    # now let us define a function that calculates the cosine similarity entries in the similarity matrix
    import pandas as pd
    import numpy as np

    def getSimilarityMatrix(vectors):
        v = np.array(list(vectors.values())).T
        sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
        keys = list(vectors.keys())
        matrix = pd.DataFrame(sim, columns = keys, index = keys)

        return matrix

    similarityMatrix = getSimilarityMatrix(allVectors)


    # 3. **Prepare top-k lists**
    from numpy.testing import assert_almost_equal
    import pickle

    k = 5 # the number of top similar images to be stored

    similarNames = pd.DataFrame(index = similarityMatrix.index, columns = range(k))
    similarValues = pd.DataFrame(index = similarityMatrix.index, columns = range(k))

    for j in tqdm(range(similarityMatrix.shape[0])):
        kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending = False).head(k)
        similarNames.iloc[j, :] = list(kSimilar.index)
        similarValues.iloc[j, :] = kSimilar.values

    similarNames.to_pickle("similarNames.pkl")
    similarValues.to_pickle("similarValues.pkl")


    # 4. **Get and visualize similar images for four example inputs**
    inputImages = ['domestic_cat_s_000907.JPEG',
                  'hydrofoil_s_000078.JPEG',
                  'sea_boat_s_001456.JPEG',
                  'jetliner_s_001705.JPEG'] # ****keys****


    numCol = 5
    numRow = 1

    def setAxes(ax, image, query = False, **kwargs):
        value = kwargs.get("value", None)
        if query:
            ax.set_xlabel("Query Image\n{0}".format(image), fontsize = 6)
        else:
            ax.set_xlabel("Similarity value {1:1.3f}\n{0}".format( image,  value), fontsize = 6)
            ax.set_xticks([])
            ax.set_yticks([])

    def getSimilarImages(image, simNames, simVals):
        if image in set(simNames.index):
            imgs = list(simNames.loc[image, :])
            vals = list(simVals.loc[image, :])
            if image in imgs:
                assert_almost_equal(max(vals), 1, decimal = 5)
                imgs.remove(image)
                vals.remove(max(vals))
            return imgs, vals
        else:
            print("'{}' Unknown image".format(image))


    # print images
    import matplotlib.pyplot as plt
    import numpy as np
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    def plotSimilarImages(image, simiarNames, similarValues):
        simImages, simValues = getSimilarImages(image, similarNames, similarValues)
        fig = plt.figure(figsize=(10, 20))

        # now plot the  most simliar images
        for j in range(0, numCol*numRow):
            ax = []
            if j == 0:
                img = Image.open(os.path.join(inputDir, image))
                ax = fig.add_subplot(numRow, numCol, 1)
                setAxes(ax, image, query = True)
            else:
                img = Image.open(os.path.join(inputDir, simImages[j-1]))
                ax.append(fig.add_subplot(numRow, numCol, j+1))
                setAxes(ax[-1], simImages[j-1], value = simValues[j-1])
            img = img.convert('RGB')
            plt.imshow(img)
            img.close()

        plt.show()


    inputDir = InputDir+str(i)
    image = inputImages[i]
    plotSimilarImages(image, similarNames, similarValues)
  
if __name__ == '__main__':
  app.run(main)
