# *** Training a Classifier ***
# Using torchvision, it’s extremely easy to load CIFAR10.
# __________________________________________________________________________________

# Object Classifier
# Data labeling with image classification

class predictor():
    def __init__(self, path, Model, Pickle): 
        
        self.path = path
        self.Model = Model
        self.Pickle = Pickle
        
        
    def get_frame(self):
        
        import torch
        import sys, os
        import torchvision
        import torchvision.transforms as transforms
        from six.moves import cPickle as pickle
        from absl import app
        from absl import flags
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        import cv2
        
        # 1. The output of torchvision datasets are PILImage images of range [0, 1].
        #    We transform them to Tensors of normalized range [-1, 1].  
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root= self.path, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root= self.path, train=False,
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                    shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        # 2. Define a Convolution Neural Network:
        #     Copy the neural network from the Neural Networks section before and 
        #     modify it to take 3-channel images (instead of 1-channel images as it was defined).

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            
        net = Net()
        

        # 3. Load pre-trained
        PATH = self.Model
        net.load_state_dict(torch.load(PATH))
        
        # 4. Test the network on the test data: 
        # Let us display an image from the test set to get familiar.

        # filenames
        def unpickle(file):
            fo = open(file, 'rb')
            dict = pickle.load(fo, encoding ='latin1')
            X = dict['filenames'][:4]
            fo.close()
            return X

        F = unpickle(self.Pickle)
        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        import matplotlib.pyplot as plt
        import numpy as np
        
        def imshow(img):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
        #imshow(torchvision.utils.make_grid(images))
        # print('GroundTruth: ', ' '.join('%s' % F[j]+':'+classes[labels[j]] for j in range(4)))

        

        # 5. Preddict:
        # We have trained the network for 2 passes over the training dataset. 
        # But we need to check if the network has learnt anything at all.
        # We will check this by predicting the class label that the neural network outputs, 
        # and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.

        pred_dict = {}
        for i in range(len(F)):
            pred_dict[str(F[i])] = classes[labels[i]]
        
        return pred_dict



# _____________________________________________________________

# Tranining Network the data labeling with image classification 

# _____________________________________________________________ 

class Classifier(object):
    def predictor(self, Path, Model, Pickle, Epoch):
        
        self.Path = Path
        self.Model = Model
        self.Pickle = Pickle
        self.Epoch = Epoch
         
        import torch
        import sys, os
        import torchvision
        import torchvision.transforms as transforms
        from six.moves import cPickle as pickle
        from absl import app
        from absl import flags
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        import cv2
        # 1. The output of torchvision datasets are PILImage images of range [0, 1].
        #    We transform them to Tensors of normalized range [-1, 1].  
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=self.Path, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=self.Path, train=False,
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                    shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        # 2. Define a Convolution Neural Network:
        #     Copy the neural network from the Neural Networks section before and 
        #     modify it to take 3-channel images (instead of 1-channel images as it was defined).

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            
        net = Net()
        
        # 3. Define a Loss function and optimizer:
        # Let's use a Classification Cross-Entropy loss and SGD with momentum.
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



        # 4. Train the network:
        # This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.
        
        for epoch in range(self.Epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    
        print('Finished Training')

        # save & load
        PATH = self.Model
        # torch.save(net.state_dict(), PATH)# reloadnet = Net()
        net.load_state_dict(torch.load(PATH))
        
        # 5. Test the network on the test data: 
        # Let us display an image from the test set to get familiar.

        # filenames
        def unpickle(file):
            fo = open(file, 'rb')
            dict = pickle.load(fo, encoding ='latin1')
            X = dict['filenames'][:4]
            fo.close()
            return X

        F = unpickle(self.Pickle)
        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        import matplotlib.pyplot as plt
        import numpy as np
        
        def imshow(img):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%s' % F[j]+':'+classes[labels[j]] for j in range(4)))

        

        # 6. Preddict:
        # We have trained the network for 2 passes over the training dataset. 
        # But we need to check if the network has learnt anything at all.
        # We will check this by predicting the class label that the neural network outputs, 
        # and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.

        pred_dict = {}
        for i in range(len(F)):
            pred_dict[str(F[i])] = classes[labels[i]]
            
        print(pred_dict)
# *************    
if __name__ == '__main__':
    Classifier().predictor('./data', './checkpoint/cifar_net.pth', './data/data/cifar-10-batches-py/test_batch', 0)

        