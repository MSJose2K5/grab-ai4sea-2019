# Grab AI for S.E.A. 2019

## The Computer Vision Challenge
**How might we automate the process of recognizing the details of the vehicles from images, including make and model?**

This is a data science assignment where you are expected to create a data model from a given training dataset.

## PROBLEM STATEMENT
**Given a dataset of distinct car images, can you automatically recognize the car model and make?**

You can use the ["Cars"](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset and the ["Training"](http://imagenet.stanford.edu/internal/car196/cars_train.tgz) dataset for this challenge.

You are expected to create a Data Model based on the ["Cars"](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset in order to solve the problem statement(s).

You should also provide step by step documentation on how to run your code. Our evaluators will be running your data models on a test dataset. 

## JUDGING CRITERIA
From the ***Model Performance*** section:

Note that your model must **output a confidence score for every classification**.
 
Submissions will be **evaluated by accuracy, precision and recall**. Given that several solutions have been published for this problem before, we recommend you emphasise how your solution differentiates, for example, according to the other listed evaluation criteria (originality, code quality, etc).

## Solution

## tl;dr: Model Performance
In the last 6 runs using the Parameters/Hyper-Parameters and [ResNet152](https://pytorch.org/docs/stable/torchvision/models.html) torchvision model (as listed in the notebook), and the environment in Part 6, the test accuracies achieved are as follows:

### Test Accuracy: **92.39%, 92.39%, 92.51%, 92.54%, 92.54%, 92.55%**

According to [*Dehghan et al., 2017*] (see Citation #4 at the bottom), the [**Top-1 car classification accuracy on Stanford car dataset**](https://www.sighthound.com/technology/vehicle-recognition/benchmarks) is reproduced in the table below:

    |                        |  Top-1   |    
    | Methods                | Accuracy |
    | ---------------------- | -------- |
    | Sighthound [See Note]  |   93.6%  |
    | Krauze et al. [2]      |   92.8%  |
    | Lin et al.    [3]      |   91.3%  |
    | Zhang et al. (2016)    |   88.4%  |
    | Xie et al.   (2015)    |   86.3%  |
    | Gosselin et al. (2014) |   82.7%  |

Based on this [Vehicle Recognition Benchmarks](https://www.sighthound.com/technology/vehicle-recognition/benchmarks) as cited by [*Dehghan et al., 2017*], the highest test accuracy of **92.55%** achieved in this solution is ***only 0.27% lower*** than that achieved in the original [*Krauze et al., 2015*] paper. 

### Note on Sighthound
Sighthound's [Computer Vision Solutions](https://www.sighthound.com/products/sighthound-io) are ***not*** open-source.

## Basis

The solution is loosely based on, but is ***a heavily-modified version of***, 
[fast.ai's Practical Deep Learning for Coders, v3: Lesson 1 - ImageClassification.](https://course.fast.ai/videos/?lesson=1)
This fast.ai [Lesson 1](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb) uses the [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by O. M. Parkhi et al., 2012 which features 12 cat breeds and 25 dogs breeds. In contrast, the **Stanford Cars Dataset** has 196 classes.

## Packages Used (and Version)
      Package      Version
    . python       [3.6.8]       
    . numpy        [1.16.2]      
    . pandas       [0.24.2]      
    . matplotlib   [3.0.3]       
    . scipy        [1.2.1]       
    . opencv       [4.1.0]       
    . fastai       [1.0.53.post2]
    . pytorch      [1.1.0]       
    . torchvision  [0.2.2]       

## Execution Time and OS Environment
On an Asus ROG G752VS with QuadCore Intel Core 2.70GHz i7-6820HK, 48GB RAM and 8GB GTX 1070, running Windows 10 Pro (Version: 1809, Build: 17763.557), this solution runs in **5.5-6.0 hours**.

## Dataset

The [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) Overview states:

"The *Cars* dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of *Make, Model, Year*, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe."

From the link above, download the [training images - **cars_train.tgz**](http://imagenet.stanford.edu/internal/car196/cars_train.tgz), [testing images - **cars_test.tgz**](http://imagenet.stanford.edu/internal/car196/cars_test.tgz), [devkit - **car_devkit.tgz**](https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz), and [test annotations - **cars\\_test\\_annos\_withlabels.mat**](http://imagenet.stanford.edu/internal/car196/cars_test_annoswithlabels.mat). Place these 4 files in the **DATA_PATH** folder, which in our case is **'./data/'**.

## Folder Structure
If you followed the dataset download instruction above, your project folder structure should look similar to this [beginning folder structure](https://github.com/MSJose2K5/grab-ai4sea-2019/blob/master/folder_structure.jpg). (This is how it looks in Windows 10.)

    D:\grab-ai4sea-2019
    |--- data
    |    |--- car_devkit.tgz
    |    |--- cars_test.tgz
    |    |--- cars_test_annos_withlabels.mat
    |    |--- cars_train.tgz
    |--- 98-test-conda-setup-v19-0615-pt11.ipynb
    |--- folder_structure.jpg
    |--- gaiforsea-cv-FINAL.ipynb
    |--- README.md


## Approach/Workflow

### Train/Valid Split
We will split the Train data (8,144 images) using a **90%/10%** ratio, resulting in the following: 

    Train: 7,330 Files; Valid: 814 Files

I tried ratios of 80%/20% and 85%/15% but the 90%/10% chosen above produced the best results.

### Model
I tried both DenseNet161 and ResNet152, doing dozens of training runs on each. The highest accuracy I got for DenseNet161 was 92.19% (range: 91.78-92.19) and for ResNet152 is 92.55% (range: 91.97-92.55). I settled for **ResNet152**, not only because it produced the highest accuracy but it also produced consistently high (92.39+) accuracies. 

### Metrics
Here I chose [[accuracy,top\\_k_accuracy]](https://docs.fast.ai/metrics.html). We can thus see the following training progress statistics/metric per epoch:

    epoch  train_loss  valid_loss  accuracy  top_k_accuracy  time

### Image Size and Batch Size
We will use an image size of **224x224** for our ResNet152 model. As noted in the official [TORCHVISION.MODELS](https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models) page, all the models have a default image size of 224x224, *except* Inception v3.

    Important: In contrast to the other models the inception_v3 expects tensors with a size of N x 3 x 299 x 299, so ensure your images are sized accordingly.

I chose a batch size of **16**, primarily because of GPU memory limitation. On my 8GB GTX 1070 GPU, I always get an out-of-memory CUDA error whenever I tried a batch size > 20. (I did try a batch size of 20, but i didn't notice any increase in the accuracy.)

It's certainly worth a try in the future to do a batch size of 32 or even 64 ***and/or*** an image size of 299 or 448.

### Data Batches
Here we utilize fast.ai's [**ImageDataBunch**](https://docs.fast.ai/vision.data.html#ImageDataBunch) object, as follows:

    data = ImageDataBunch.from_folder(DATA_PATH, 'train', 'valid', 
                                  ds_tfms=get_transforms(do_flip=False, flip_vert=True, max_rotate=MAX_ROT), 
                                  size=IMG_SIZE, num_workers=N_WORKERS, bs=B_SIZE).normalize(imagenet_stats)

For the [**get_transforms()**](https://docs.fast.ai/vision.transform.html#get_transforms) method, I left most attributes in their default values *except* do\_flip, flip\_vert, and max\_rotate. For **max\_rotate** I changed the value from the *default* 10.0 to 5.0 which produced better results. (See **MAX_ROT** variable.)

### Training - Initial
Finally we train our model using fast.ai's [**cnn_learner**](https://docs.fast.ai/vision.learner.html#cnn_learner) using **30** epochs. During the initial training, all layer groups in the model *except* the last are *untrainable* (i.e. requires_grad=False).

Looking at the **Loss vs. Batches processed** graph produced by the [**recorder.plot_losses()**](https://docs.fast.ai/basic_train.html#Recorder.plot_losses) method, we see that the Train and Validation Loss curves are OK without any overfitting.

### Training - Final
Once we determine that the model is working, we use fast.ai's [**Learning Rate Finder**](https://docs.fast.ai/callbacks.lr_finder.html). As the official docs state, "Learning rate finder plots lr vs loss relationship for a [Learner](https://docs.fast.ai/basic_train.html#Learner). The idea is to reduce the amount of guesswork on picking a good starting learning rate."

After a lot of experimentation with the dataset, I found that a **max_lr=slice(1e-5, 1e-3)** value works best.

We then [**unfreeze**](https://docs.fast.ai/basic_train.html#Learner.freeze) the model, setting every layer group to *trainable* (i.e. requires_grad=True), and do our final training using **30** epochs again.

### Predictions
When making predictions on the Test dataset (Part 3 in the solution notebook), we make use of the test annotations file "**cars\_test\_annos\_withlabels.mat**".

We apply "Test Time Augmentation" using fast.ai's [**learn.TTA()**](https://docs.fast.ai/basic_train.html#_TTA) method.

Then we get the test accuracy with the following line:

    acc = accuracy(preds, y)

### Confidence Score
As stated in the ***Model Performance*** section, "your model must **output a confidence score for every classification**".

This means that the output file should contain T rows (corresponding to the number of testing images) with each row containing 196 entries/probabilities (corresponding to the number of classes) **totaling 1.0**.

This file, **preds\_proba\_ALL.csv**, is generated in Part 4 of the solution notebook.

### Precision and Recall
As stated in the ***Model Performance*** section, "submissions will be **evaluated by accuracy, precision and recall**".

**Accuracy** is computed in Part 3 of the solution notebook.

**Precision/Recall** is computed in Part 5 of the solution notebook.


## Citation(s): 
**[1] 3D Object Representations for Fine-Grained Categorization**. [PDF here](http://vision.stanford.edu/pdf/3drr13.pdf)

    Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
    4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.
       
**[2] Fine-grained recognition without part annotations**. [PDF here](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Krause_Fine-Grained_Recognition_Without_2015_CVPR_paper.pdf)

    Jonathan Krause, Hailin Jin, Jianchao Yang, Li Fei-Fei
    2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Boston, MA, USA. June 7-12, 2015.
       
**[3] Bilinear CNN Models for Fine-grained Visual Recognition**. [PDF here](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf)

    Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji
    2015 IEEE International Conference on Computer Vision (ICCV). Santiago, Chile. Dec. 7-13, 2015.
       
**[4] View Independent Vehicle Make, Model and Color Recognition Using Convolutional Neural Network**. [PDF here](https://arxiv.org/pdf/1702.01721)

    Afshin Dehghan, Syed Zain Masood, Guang Shu, Enrique G. Ortiz
    arXiv:1702.01721v1 [cs.CV]. Winter Park, FL, USA. Feb. 6, 2017.
       