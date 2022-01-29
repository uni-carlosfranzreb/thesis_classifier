# Classifier

Here, a classifier model will be trained with the documents of OpenAlex. The model should account for the noise of the subjects (accuracy is at around 80 %).

One of the main advantages of this approach is that the accuracy of the model can always be improved by feeding it more documents from OpenAlex.

The architecture and embedding choices are similar to those described in Giargulo's paper:

1. pre-trained fasttext word vectors, trained on a Wikipedia dump of 2017 and three other sources,
2. two convolution layers with ReLU activations, followed by max. pooling layers,
3. a fully connected two layer neural network, whose output layer has as many nodes as there are subjects,
4. training occurs with batch size 10, learning rate 0.1, momentum 0.5 and Nesterov accelerated gradient.

We also ensure labels are coherent, as Giargulo: all ancestors of assigned labels should also be assigned.

However, instead of using sigmoid cross entropy as a loss function, we use Ben-Baruch's assymetric loss, which accounts for the imbalance between positive and negative labels and considers noise. The loss dynamically down-weights and hard-thresholds easy negative samples, while also discarding possibly mislabeled samples.