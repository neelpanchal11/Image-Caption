
# Image-Captioning
CNN-Encoder and RNN-Decoder (Bahdanau Attention) for image caption or image to text on [MS-COCO](http://cocodataset.org/#home) dataset.

## Task Description 
Given an image like the example below, our goal is to generate a caption such as "a surfer riding on a wave".

![Man Surfing](https://tensorflow.org/images/surf.jpg)

To accomplish this, you'll use an attention-based model, which enables us to see what parts of the image the model focuses on as it generates a caption.

![Prediction](https://tensorflow.org/images/imcap_prediction.png)

The model architecture is similar to [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).

## Main principle

The model consists of CNN-Encoder and RNN-Decoder. The CNN-Encoder is used to extract the information of the input image to generate the intermediate representation H, and then use RNN-Decode to gradually decode the H (using Bahdanau Attention) to generate a text description corresponding to the image.


```
Input: image_features.shape (16, 64, 2048)
---------------Pass by cnn_encoder---------------
Output: image_features_encoder.shape (16, 64, 256)

Input: batch_words.shape (16, 1)
Input: rnn state shape (16, 512)
---------------Pass by rnn_decoder---------------
Output: out_batch_words.shape (16, 5031)
Output: out_state.shape (16, 512)
Output: attention_weights.shape (16, 64, 1)
```

## Code test pass
+ Pyhon 3.6
+ TensorFlow version 2

