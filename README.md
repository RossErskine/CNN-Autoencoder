# Convolutional Autoencoder for detecting outlier in the Sloan Digital Sky Survey



### Ross Erskine
  
## Introduction
The Sloan Digital Sky Survey (SDSS) is a international collaboration that has mappped up-to one-third of the night sky with over three million observations  of Galaxies, Stars,and Quasars with multi-colour imagary, over the last 20 years. With the next phase SDSS-V on the horizon with the next data release in December 2022, researchers will want to exlore more and more data. The discovery of outliers proposes new opportunities in exploring unknown phenomena and can produce some of the weirdest Galaxies in the universe , helping us understand more and more about what can possibly be out there. Convolution autoencoder (CAE) architecture is generally used as a dimensionality reduction technique or feature learning. Although can also be used as a unsupervised method that can be trained to encode input features into smaller dimensional space, then reconstruct the features in to a reduced dimensional space, allowing us to measure discrepancies from the original and identify outliers.



## UML 
![UML](https://user-images.githubusercontent.com/46631932/207575709-015fab24-37eb-4faf-913a-a87e7aab5d06.png)
To run the program run the main.py file 

## Methodology
Convolutions are based on the LeNet-5 network , which a kernel or filter is applied to each colour channel extracting the most predominant features to create a new feature map. This naturally shrinks the size of the input, although padding can be added if the size of the input is to be attained .
Convolutional Autoencoders (CAE) is an unsupervised learning algorithm, which is composed of two parts an encoder and a decoder. The encoder $h=f(x)$ compresses the input into a smaller dimensional latent space representation or hidden layer $h$ using convolutions, which extracts the most relevant features. The decoder $g(f(x))=x'$ decompresses that latent space representation back up using convolutions into the original image. The CAE algorithm learns by maximising the information and minimising the reconstruction error or loss function $L(x, g((f(x)))$ which in this case is the mean-squared error Equation below. 

$$\text{MSE}= \frac{1}{n}\sum^n_{i=1}(\hat{y}_i - y_i)^2$$
The images used from galaxyZoo could have been up to $450x450$ diameter we opted for an input size of $128x128x3$ due to memory restrictions. We also decided to split our CAE into two seperate classes as we would like to use the encoder part alone to create our KDE later on; the two version where based on work from  Each kernel/filter used is $3x3$ with a stride of $2$, the first layer on the encoder using convolutions outputs a dimension $64x64x32$, layer 2 output is $32x32x16$, layer 3 output is $16x16x8$ and finally the fully compressed image resides at $8x8x4$ giving a reduced total dimensionality with $256$ total parameters. The decoder is a mirror of the encoder that starts with input of $8x8x4$ onto the second layer with output of $16x16x8$ and so on until we get to our original dimensions of $128x128x3$. 

The original galaxy zoo images from galaxtZoo has up to 250 thousand images we are only going to use 20 thousand for this experiment, training will be done in batches of 64 and 1000 epochs, with a validation set  of batches of 32 to see whether the generalisation gap is minimal. This will ensure that the reduced dimensionality will have the best chance to represent our model.

### Pipeline of KDE creation.
![CAE_KDE drawio](https://user-images.githubusercontent.com/46631932/207576936-8507e6a6-acc3-4376-8ee4-a70eaebb7377.png)


After our training on the CAE has finished we will then attempt to create our PDF from a single batch from the training data set.  above shows the pipeline where we use just the encoder part of our CAE by passing each image from a single batch through the encoder we achieve our reduced latent space representation which was dimension $8x8x4$, then we flatten to a single vector of $256$ length and using a KDE Equation below; where $\alpha$ is constant normalisation factor and $h$ is the kernel bandwidth  this produces or smoothed over PDF, which we can then take our thresholds from tail end of the distribution. This allows us to then test our model with either a single image or a batch of images, to classify if our model believes  that the image is an outlier or not. 

$$\overline{f}_X(x) = \frac{1}{\alpha} \sum^n_{i=1} K\left(\frac{x-x_i}{h}\right)$$


## Results 
The training loss see below seem to suggest that the model converged after about 500 epochs meaning we probably did not need as many as a 1000 epochs to train, nothing about the training suggests that we would need more than 20000 images to train, however more images would take longer to converge perhaps giving better results. 

### Training loss
![training_loss](https://user-images.githubusercontent.com/46631932/207577279-a80f5fa3-dda9-4fc4-bf38-3171abdbed18.png)


After testing our data on a sample batch from the validation set of data not seen during training. The results from the batch of an average density of -9727, standard deviation density of 11755, an average reconstruction error of 0.0007 and a standard deviation reconstruction error of 0.0003. Our model seem to suggest out of the sample batch 32 images that 4 of them where outliers see below. However nothing visually stands out such as lines from asteroids or satellite's or very bright objects or merging galaxies , Although according to the encoded data from the validation batch sample see  that you could believe that 4 objects stand out as outliers. 

### Batch scatter plot
![sample_batch_plot](https://user-images.githubusercontent.com/46631932/207577336-95412f75-0a75-477f-9e41-b05c16f021c1.png)


Looking at further work: time restraints meant the model was trained on only on 20000 images this could be increased give a better training and better representation of encoding, also more layers could be added such as dense layers, linear layers , or max-pooling layers \cite{bhattiprolu_python_2022}. However, outlier detection does seem to have a improved results when using spectra rather than actual image , \cite{wei_mining_2013}, . Also increased tests perhaps with known outliers such as Boyajian's star  to help understand the CAE model more, and more analysis between different densities obtained from the model.

## Conclusion
We proposed a method of using Convolutional autoencoder CAE to find outliers in the SDSS galaxyZoo dataset by exploiting the reduced dimensional space created by the encoder part of the CAE. We used a method of KDE that smooths over the distributions of samples to create a PDF so we could estimate outliers based on only a sample of the data. Although the model trained well and gave us some results of 4 outliers from a batch of 64, the outliers did not stand out as significant. we need more rigorous testing to allow us to understand the model further such as increased batch sizes, tests on known outliers and more analysis from multiple densities.

## Plots and diagrams
### Original and reconstructed images
![Orig](https://user-images.githubusercontent.com/46631932/207577115-08611923-4445-4f16-9251-1915abe4dfd0.png)

![Recon](https://user-images.githubusercontent.com/46631932/207577148-9f58e37e-f6c8-4eca-9000-6189e91b44c9.png)

### Outliers found
![outlier](https://user-images.githubusercontent.com/46631932/207577227-16655a43-2c5e-4274-8efb-f79de0b46098.png)












