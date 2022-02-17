# Classifier

Here, a classifier model will be trained with the documents of OpenAlex. The model should account for the noise of the subjects (accuracy is at around 80 %).

One of the main advantages of this approach is that the accuracy of the model can always be improved by feeding it more documents from OpenAlex.


## Architecture

The architecture and embedding choices are similar to those described in Giargulo's paper:

1. pre-trained fasttext word vectors, trained on a Wikipedia dump of 2017 and three other sources,
2. two convolution layers with ReLU activations, followed by max. pooling layers and finally a dropout layer,
3. a fully connected two layer neural network, whose output layer has as many nodes as there are subjects,
4. training occurs with batch size 10, learning rate 0.1, momentum 0.5 and Nesterov accelerated gradients.

He does not mention clipping in the paper, so we will first try to train the model without.

An issue with Giargiulo implementation is that the first max-pooling (MP) layer is said to have kernel size 1. This can not be right, as the input would remain unchanged. It wouldn't make sense either to be a MP layer of size 2, as the resulting output would have as many vectors are as expected from the second convolutional layer. In numbers: thei first conv. layer outputs 200 vectors; the second one, 100 vectors. If there was a pooling layer with kernel size 2 in between, the second convolutional layer would not reduce the dimensionality of the input. This is possible but unlikely (is it?). We therefore remove this pooling layer.

The convolution layers are stated to be one-dimensional. The pooling layers are not. In NLP, the common pooling practice is [max-pooling over time](https://cezannec.github.io/CNN_Text_Classification/). This (blog post)[https://lena-voita.github.io/nlp_course/models/convolutional.html] explains hwo to use convolutions and pooling for text very well:

* Convolutions (when padded) reduce the dimensionality of the vectors. A one-dimensional convolution receive _k_ one-dimensional vectors of size _n_ as input and output a single one-dimensional vector of size _m_. _k_ is the kernel size and _m_ is the number of filters (what PyTorch refers to as _output channels_). Convolutions extract features from multiple word vectors at a time. Therefore, the result doesn't necessarily change the number of vectors; it affects the number of dimensions. The number of extracted features is the deciding factor here. The number of vectors can be ensured to remain constant with padding.
* Max-pooling extracts the most salient features across each dimension. Therefore, the number of dimensions remains the same; the number of vectors is what changes. For instance, if the kernel size is set to 2, the number of vectors is halved.

We also ensure labels are coherent, as Giargiulo: all ancestors of assigned labels should also be assigned.

Giargulo trains the model with the sigmoid cross entropy loss function. This is the same as the Binary Cross Entropy Loss of PyTorch.

We will train it again with Ben-Baruch's assymetric loss, which accounts for the imbalance between positive and negative labels and considers noise. The loss dynamically down-weights and hard-thresholds easy negative samples, while also discarding possibly mislabeled samples.

## Word embeddings

The fasttext file (`wiki-news-300d-1M-subword.vec`) includes 999,994 300-dimensional vectors for words without lemmatizing or even lower-casing. For instance, all these words are in the file: `machine, machines, learn, learning, learns, learned`, both lower- and upper-cased. Although Mikolov mentions training phrases as well, they were not published together with the words.

Given that we are only interested in lower-cased words, we create a new file containing only lower-cased words. Upper-cased words that don't have lower-cased equivalents are lower-cased and added to the file. This procedure yields a file with vectors for 000 words.

## Training

Giargulo splits the dataset into 99 % training and 1 % test sets. This is done by the file `split_data.py`.

## Structure of the data file

Data files must follow a certain format. There can be many of them, as the Dataset receives the folder as an input.

Data is ordered by subject, as it was extracted ensuring that all subjects are present. Therefore, the first level of the file is the subject IDs. Each ID is mapped to a list of documents, containing text and subjects of the documents. The data is the concatenation of the title and the abstract. The subjects are extracted from the OpenAlex API with the file `get_publications.py`.

If the data is still raw text, you can process it with the file `process_docs.py`. It will tokenize and lemmatize the text, without removing any words.

Here is an example:

```json
[
  {
    "data": "text (str)",
    "subjects": "list of subject IDs"
  }
]
```

## Open questions

* Subjects are assigned to documents with proability scores. How can I include these scores? Gargulio's subjects are assigned by humans (no scores).
* How are the scores of MAG's subject assignments computed? Are they probabilities?

## Training diary

Here I will document what happens in each training procedure.

### 1643821400

Here I implemented Giargulo's training. The test loss decreased slowly but steadily. I decided to stop the training after 10 epochs, as it had been already training for 10 epochs. There is still room for improvement. If the results are promising, I may train it further.

### 1643876999

Here I used the same parameters as above, only changing the loss function. I used Ben-Baruch's asymmetric loss. I have stopped the training in the middle of the fifth epoch because something weird was happening. The results were being exactly the same for every epoch except the first one. Model parameters were not being updated. Maybe it is because of exploding gradients?

The test losses are exactly the same for all epochs, so maybe something weird happened while evaluating the model, like deactivating the backward pass of the loss function. Now I will use the BCE loss function for all tests, for the sake of comparison. I will let it run again with this loss function.

Then, I will implement Ben-Baruch's training procedure, and consider introducing gradient clipping.

### 1643903291

Again, all test losses remained the same throughout the six epochs.

### 1643982969

Now the model was trained correctly. However, it started overfitting very early. I have increased the dropout probability by one order of magnitude (from .001 to .01), to help the model generalize better.

It also ran out of scheduler steps and stopped training during the fifth epoch. To prevent this, I have tripled the number of steps.

### 1644052071

This model avoided overfitting much better. The training loss increased after the second epoch, whereas the testing loss steadily decreased. The dropout will be increased again in the next run.

### 1644093287

This model, with increased dropout rate (from 0.01 to 0.05), was extremely similar to the previous one. I have now increased the batch size to 32 and decreased the scheduler steps from 3,000 to 2,000, to account for the smaller number of batches.

### 1644314094

The test loss decreased monotonically, but very slowly. As Garguilo did, I didn't use a scheduler. This is not optimal. I will try a couple of schedulers now to accelerate training.

### ASL

The models trained with ASL were not working properly because the original ASL implementation sums the individual label losses, instead of averaging them, like PyTorch's BCE does.

## TODO

1. Answer open questions
2. Train models with less words I.e. look how many words we usually have. Medical data usually has more and 400 may be too much for our dataset.
3. Sum vectors of words to represent each doc and train a model with that sum.
