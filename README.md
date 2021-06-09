# BIRD-CLASSIFICATION

# ANN-CNNForImageClassification
CNN run on 52,000 images of birds for bird recognition in images

1.	Dataset used for the homework is a combination of CIFAR10 dataset and the Caltech UCSD birds dataset. The model predicts if an image contains a bird or not. There is a total of 52,000 images the network uses to train itself. The link for the combined dataset represented as a .pkl file is given below.

[dataset.zip](https://s3-us-west-2.amazonaws.com/ml-is-fun/data.zip)

2.	The data is unpickled, shuffled to move images from its own class to different positions so that when I split data into train and test, both splits have all kinds of images.

```
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"), encoding="bytes")
X, Y = shuffle(X, Y)
```

3.	The network shape is 32x32 image with RGB i.e. 3 channels

```
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
```

4.	The architecture of the model consists of three 2D convolutional layers and between them, maxpooling. There is then a fully connected layer after which I am dropping out half the data to prevent over fitting. This is then finally followed by a softmax layer. Convolutions break down an image into overlapping tiles. Each tile is processed as an image after which maxpool layer reduces the output array size from the convolution layer to keep only the important features got from the image tiles.  Fully connected softmax network is used to make predictions i.e. if the image is a bird or not. 

```
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
```

5.	The network is trained using the below parameters

```
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
```

6.	Model is fit with the below conditions. At the end of 100 epochs, an accuracy of 91.90% is attained. Model takes about half a day to train.

```
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='bird-classifier')
```

# Prediction with Random Forests

```
In this chapter, we're going to look at classification techniques with random forests. We're
going to use scikit-learn, just like we did in the previous chapter. We're going to look at
examples of predicting bird species from descriptive attributes and then use a confusion
matrix on them.
Here's a detailed list of the topics:
Classification and techniques for evaluation
Predicting bird species with random forests
Confusion matrix
```

Random forests

```
Random forests are extensions of decision trees and are a kind of ensemble method.
Ensemble methods can achieve high accuracy by building several classifiers and running a
each one independently. When a classifier makes a decision, you can make use of the most
common and the average decision. If we use the most common method, it is called voting.
```

# Predicting bird species with random forests

Here we will be using random forests to predict a bird's species. We will use the Caltech- UC San Diego dataset (http://www.vision.caltech.edu/visipedia/CUB–2OO–2Oll.html), which contains about 12,000 photos of birds from 200 different species. Here we are not going to look at the pictures because that would need a convolutional neural network (CNN) and this will be covered in later chapters. CNNs can handle pictures much better than a random forest. Instead, we will be using attributes of the birds such as size, shape, and color.

[data.zip](https://deepai.org/dataset/cub-200-2011)



Making a confusion matrix for the data

Let's make a confusion matrix to see which birds the dataset confuses. The confusion_matrix function from scikit-learn will produce the matrix, but it's a pretty big


Since the bird's names are sorted, lesser is the square of confusion. Let's compare this with the simple decision tree:

![decision_tree](https://github.com/saddy8838/BIRD-CLASSIFICATION/blob/main/decision%20tree.jpg)


Here, the accuracy is 27%, which is less than the previous 44% accuracy. Therefore, the decision tree is worse. If we use a Support Vector Machine (SVM), which is the neural network approach, the output is 29%:


![svm](https://github.com/saddy8838/BIRD-CLASSIFICATION/blob/main/Support%20Vector%20Machine%20(svm).jpg)



The random forest is still better.


Let's perform cross-validation to make sure that we split the training test in different ways. The output is still 44% for the random forest, 25% for our decision tree, and 27% for SVM, as shown in the following screenshot:



![compare](https://github.com/saddy8838/BIRD-CLASSIFICATION/blob/main/comparing%20image.jpg)


The best results are reflected through random forests since we had some options and questions with random forests.
For example, how many different questions can each tree ask? How many attributes does it look at, and how many trees are there? Well, there are a lot of parameters to look through, so let's just make a loop and try them all:

![random_forest](https://github.com/saddy8838/BIRD-CLASSIFICATION/blob/main/random%20forest.jpg)

These are all the accuracies, but it would be better to visualize this in a graph, as shown here:

![campare_img](https://github.com/saddy8838/BIRD-CLASSIFICATION/blob/main/comparing%20image.jpg)

We can see that increasing the number of trees produces a better outcome. Also, increasing the number of features produces better outcomes if you are able to see more features, but ultimately, if you're at about 20 to 30 features and you have about 75 to 100 trees, that's about as good as you're going to get an accuracy of 45%.



References:

[Medium](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721)


