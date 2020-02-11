# Should you ever use a one-step forecaster?¶

In this notebook, I will show, how beneficial can be to use multiple forecast steps even if you want to forecast only one step ahead with neural networks. The models are built with Tensorflow. I used a simple model which has conv1d and LSTM layers. The forecasted time-series data is synthetic because I wanted to ensure that there exists a real pattern, but I tried to make it hard enough for the models to learn.

I compared the models' performance based only on the first forecast step. One model outputs only the next forecast step, the other model forecasts 10 steps, but we are interested only in the first step.

I found that the 10 step model is better to forecast the first step, than the other model which focused on the first step alone.

I used a sequence-to-sequence model. Aurélien Géron mentions in his book (Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow) that the advantage of seq-to-seq is that more error gradient will both stabilize and speed up training.

The idea of using more forecast steps than needed can be beneficial as well. It will result in slower training but can have a regularizer effect, especially in case of a time series with high variance. I checked only one time series and one model architecture so far, so I can't say that this can help in most cases, but in some cases, it can improve the forecast.

notebook: https://github.com/sinusgamma/multitarget_training/blob/master/multitarget_conv_lstm.ipynb
