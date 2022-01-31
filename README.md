# Classifier

Here, a classifier model will be trained with the documents of OpenAlex. The model should account for the noise of the subjects (accuracy is at around 80 %).

One of the main advantages of this approach is that the accuracy of the model can always be improved by feeding it more documents from OpenAlex.


## Architecture

The architecture and embedding choices are similar to those described in Giargulo's paper:

1. pre-trained fasttext word vectors, trained on a Wikipedia dump of 2017 and three other sources,
2. two convolution layers with ReLU activations, followed by max. pooling layers and finally a dropout layer,
3. a fully connected two layer neural network, whose output layer has as many nodes as there are subjects,
4. training occurs with batch size 10, learning rate 0.1, momentum 0.5 and Nesterov accelerated gradient.

He does not mention clipping in the paper, so we will first try to train the model without.

An issue with Giargiulo implementation is that the first max-pooling (MP) layer is said to have kernel size 1. This can not be right, as the input would remain unchanged. It wouldn't make sense either to be a MP layer of size 2, as the resulting output would have as many vectors are as expected from the second convolutional layer. In numbers: thei first conv. layer outputs 200 vectors; the second one, 100 vectors. If there was a pooling layer with kernel size 2 in between, the second convolutional layer would not reduce the dimensionality of the input. This is possible but unlikely (is it?). We therefore remove this pooling layer.

The convolution layers are stated to be one-dimensional. The pooling layers are not. In NLP, the common pooling practice is [max-pooling over time](https://cezannec.github.io/CNN_Text_Classification/). This (blog post)[https://lena-voita.github.io/nlp_course/models/convolutional.html] explains hwo to use convolutions and pooling for text very well:

* Convolutions (when padded) reduce the dimensionality of the vectors. A one-dimensional convolution receive _k_ one-dimensional vectors of size _n_ as input and output a single one-dimensional vector of size _m_. _k_ is the kernel size and _m_ is the number of filters (what PyTorch refers to as _output channels_). Convolutions extract features from multiple word vectors at a time. Therefore, the result doesn't necessarily change the number of vectors; it affects the number of dimensions. The number of extracted features is the deciding factor here. The number of vectors can be ensured to remain constant with padding.
* Max-pooling extracts the most salient features across each dimension. Therefore, the number of dimensions remains the same; the number of vectors is what changes. For instance, if the kernel size is set to 2, the number of vectors is halved.

We also ensure labels are coherent, as Giargiulo: all ancestors of assigned labels should also be assigned.

However, instead of using sigmoid cross entropy as a loss function, we use Ben-Baruch's assymetric loss, which accounts for the imbalance between positive and negative labels and considers noise. The loss dynamically down-weights and hard-thresholds easy negative samples, while also discarding possibly mislabeled samples.

## Word embeddings

The fasttext file (`wiki-news-300d-1M-subword.vec`) includes 999,994 300-dimensional vectors for words without lemmatizing or even lower-casing. For instance, all these words are in the file: `machine, machines, learn, learning, learns, learned`, both lower- and upper-cased. Although Mikolov mentions training phrases as well, they were not published together with the words.

Given that we are only interested in lower-cased words, we create a new file containing only lower-cased words. Removing all lower-cased words yields a file with 410,568 words.

## Structure of the data file

Data files must follow a certain format. There can be many of them, as the Dataset receives the folder as an input.

Data is ordered by subject, as it was extracted ensuring that all subjects are present. Therefore, the first level of the file is the subject IDs. Each ID is mapped to a dictionary of documents, where the document IDs are mapped to a dictionary containing their data and subjects. The data is the concatenation of the title and the abstract.

Here is an example:

```
{
  'subject ID': {
    'document ID': {
      'data': data (str),
      'subjects': subject (list)
    }
  }
}
```