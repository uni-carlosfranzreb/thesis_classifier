# Classifier

Here, a classifier model will be trained with the documents of OpenAlex. Documents and subjects are represented as the sum of their word vectors.

20 different models will be trained: one for the fields, and one for each field. Documents are first fed to the field classifier. The model gives a similarity score for each field. Then, the document is fed to the models of the three top fields.

We use the same voabulary and vector embeddings of the first approach. The vectors are not very accurate, given the limited size of our dataset, and our vocabulary only covers our dataset. This means that many words of the OpenAlex documents will not be in our vocabulary and therefore not be included in the vector representation. This is fine, as we are interested in documents that are similar to those in our repositories.

One of the main advantages of this approach is that the accuracy of the model can always be improved by feeding it more documents from OpenAlex.