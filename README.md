# AutoEncoder-Anomaly-Detection
using autoencoder for detecting anomaly in mnist dataset

## Discription
> Anomalies are defined as events that deviate from the standard, happen rarely, and don’t follow the rest of the “pattern” .The problem is only compounded by the fact that there is a massive imbalance in our class labels.To accomplish this task, an autoencoder uses two components: an encoder and a decoder.The encoder accepts the input data and compresses it into the latent-space representation. The decoder then attempts to reconstruct the input data from the latent space.When trained in an end-to-end fashion, the hidden layers of the network learn filters that are robust and even capable of denoising the input data.We then presented the autoencoder with a digit and tell it to reconstruct it as below:
>
![image](https://user-images.githubusercontent.com/53394692/111328269-5633b880-8683-11eb-9352-ce0bfa48224f.png)
>
> We would expect the autoencoder to do a really good job at reconstructing the digit, as that is exactly what the autoencoder was trained to do — and if we were to look at the MSE between the input image and the reconstructed image, we would find that it’s quite low.Let’s now suppose we presented our autoencoder with a photo of an elephant and asked it to reconstruct it:
> 
![image](https://user-images.githubusercontent.com/53394692/111328482-84b19380-8683-11eb-857f-c19382268622.png)
> Since the autoencoder has never seen an elephant before, and more to the point, was never trained to reconstruct an elephant, our MSE will be very high.If the MSE of the reconstruction is high, then we likely have an outlier.

## DATASET  
> Later in this tutorial, we’ll be training an autoencoder on the MNIST dataset. The MNIST dataset consists of digits that are 28×28 pixels with a single channel, implying that each digit is represented by 28 x 28 = 784 values.
>
## STRUCTURE of This Project
> the architecture of autoencdoer is in `pyimagesearch/convautoencoder.py` and for starting the train procedure you can run following command:
```
python train_unsupervised_autoencoder.py --dataset output/images.pickle --model output/autoencoder.model

autoencoder.model: The serialized, trained autoencoder model.
images.pickle: A serialized set of unlabeled images for us to find anomalies in.
```
> these two files will store in `output` directory.
> 
> after running this `.py` file , the result of train/validation basis on our dataset will be creating,such as below :
>
![plot](https://user-images.githubusercontent.com/53394692/111337207-24beeb00-868b-11eb-9277-d1d1351ddb25.png)
>
> Furthermore, we can look at our output `recon_vis.png` visualization file to see that our autoencoder has learned to correctly reconstruct the 1 digit from the MNIST dataset:
>
![recon_vis](https://user-images.githubusercontent.com/53394692/111337469-5e8ff180-868b-11eb-93d7-455c8ad5ea31.png)
>
> for testing and detecting anomalies in dataset,run `find_anomalies.py` such as below: 
```
python find_anomalies.py --dataset output/images.pickle --model output/autoencoder.model
```
> after running this `.py` file,you see the result as below:
> 
![output](https://user-images.githubusercontent.com/53394692/111338714-656b3400-868c-11eb-94a8-1dca2d0a9ad8.PNG)
>
> since autoencoder only learned about number `1` structure basis on our congiguration in `train_unsupervised_autoencoder.py` file,and learned the fact number `3` as anomaly,we see that in anomly list two number `1` is found as anomaly,these are the incorrect results of autoencoder.



## License
> [Anomaly detection with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/) by Adrian Rosebrock
