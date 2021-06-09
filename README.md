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

6.	Model is fit with the below conditions. At the end of 100 epochs, an accuracy of 94.97% is attained. Model takes about half a day to train.

```
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='bird-classifier')
```

7.	To predict, save the model, restore model and then pass a bird image to the model for it to make a prediction. Download any bird image from Google, save it in the same path as the notebook with the name “birdtest1.bmp”.  Resize this test image to the network’s shape i.e. to None/1,32,32,3.

```
model.save("bird-classifier.tfl")
model.load("bird-classifier.tfl")
img = scipy.ndimage.imread("birdtest1.bmp", mode="RGB")
img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
prediction = model.predict([img])
```

8.	Reducing the number of epochs reduces accuracy and even for a test image with only one clear bird, reduces true positive prediction. 

9.	Removing the third convolutional layer makes a difference in accuracy. While originally for 10 epochs is 89.67%, after removal of layer it becomes 88.68%.
a
10.	Performing the dropout before the fully connected layers also affects accuracy, accuracy reduces. It changed from 89.67% to 88.70% for 10 epochs.

References:

[Medium](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721)
