# Text Style Transfer 

I recently read an interesting paper on [Neural Style Transfer](https://arxiv.org/abs/1508.06576) and tried out an example of a torch implemtation of that paper. I was impressed at how it transfered the styles while still mantaining the original picture. 

I then had the idea to try style transfer on text, the original plan was to use a 1d CNN over both datasets and use the same model as the Neural Style Transfer paper. However, after doing more research, recurent neural networks (RNN's) and LSTM's seem to do a better job of remembering context which is important for text generation so that will be the inital path I explore. 
