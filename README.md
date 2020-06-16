This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Overview


The Convolutional Deep Neural Network is loosely based on the paper of Relation Classification via Convolutional Deep Neural Network [Daojian Zeng et al].

Implementation:

To implement the Convolution Neural network, we make use of the tf.keras.layers.Conv1D( ) and  tf.keras.layers.GlobalMaxPool1D( ) functions. The Conv1D layer is added to create a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. The activation layer is mentioned as “relu” which is then applied to the output tensor. The max-pooling is then computed over the resultant tensor. This process is carried out in three layers i.e I have stacked the convolution layers and then the feature vectors are concatenated into a single sentence-level feature vector. This is then passed through the tf.layers.Dense( ) function to get the final sentence-level feature vector.

1. Three layers of convolution kernels are defined initially.

2. Three layers of max-pooling are also defined similarly.

3. Word embedding vectors and POS embedding vectors are formed by embedding lookup of the inputs.

4. This combination of word embeddings and POS embeddings are concatenated and sent to the convolution layer 1 of Convolution DNN 

5. This is then sent to the maxpooling layer 1 of the model

6. The resulting vector is similarly sent to the convolution layer 2, max-pooling layer 2, convolution layer 3 and max-pooling layer 3 sequentially.

7. The final sentence level feature vector is obtained after the passing the concatenated resulting vector from step 6 through a decoder.

The architecture of the above mentioned model is included in the report.pdf under the section Architecture.

