# clustimage

[![Python](https://img.shields.io/pypi/pyversions/clustimage)](https://img.shields.io/pypi/pyversions/clustimage)
[![PyPI Version](https://img.shields.io/pypi/v/clustimage)](https://pypi.org/project/clustimage/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clustimage/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/clustimage.svg)](https://github.com/erdogant/clustimage/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/clustimage.svg)](https://github.com/erdogant/clustimage/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/clustimage/)
[![Downloads](https://pepy.tech/badge/clustimage/month)](https://pepy.tech/project/clustimage/month)
[![Downloads](https://pepy.tech/badge/clustimage)](https://pepy.tech/project/clustimage)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
[![DOI](https://zenodo.org/badge/423822054.svg)](https://zenodo.org/badge/latestdoi/423822054)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

The aim of ``clustimage`` is to detect natural groups or clusters of images.

Image recognition is a computer vision task for identifying and verifying objects/persons on a photograph.
We can seperate the image recognition task into the two broad tasks, namely the supervised and unsupervised task.
In case of the supervised task, we have to classify an image into a fixed number of learned categories. Most packages rely on (deep) neural networks, and try solve the problem of predicting "whats on the image".
In case of the unsupervised task, we do not depend on the fact that training data is required but we can interpret the input data and find natural groups or clusters.
However, it can be quit a breath to carefully group similar images in an unsupervised manner, or simply identify the unique images.

The aim of ``clustimage`` is to detect natural groups or clusters of images. It works using a multi-step proces of carefully pre-processing the images, extracting the features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters can be determined using well known methods suchs as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
With ``clustimage`` we aim to determine the most robust clustering by efficiently searching across the parameter and evaluation the clusters.
Besides clustering of images, the ``clustimage`` model can also be used to find the most similar images for a new unseen sample.

A schematic overview is as following:

<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/schematic_overview.png" width="1000" />
</p>


``clustimage`` overcomess the following challenges: 

    * 1. Robustly groups similar images.
    * 2. Returns the unique images.
    * 3. Finds higly similar images for a given input image.

``clustimage`` is fun because:

    * It does not require a learning proces.
    * It can group any set of images.
    * It can return only the unique() images.
    * it can find highly similar images given an input image.
    * It provided many plots to improve understanding of the feature-space and sample-sample relationships
    * It is build on core statistics, such as PCA, HOG and many more, and therefore it does not has a dependency block.
    * It works out of the box.

### Installation
* Install clustimage from PyPI (recommended). clustimage is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```bash
conda create -n env_clustimage python=3.8
conda activate env_clustimage
```

* Install from pypi
```bash
pip install -U clustimage
```  

#### Import the clustimage package
```python
from clustimage import Clustimage
```

### Example 1: Digit images.
In this example we will be using a flattened grayscale image array loaded from sklearn.
The array in NxM, where N are the samples and M the flattened raw rgb/gray image.

```python
# Load library
import matplotlib.pyplot as plt
from clustimage import Clustimage
# init
cl = Clustimage()
# Load example digit data
X = cl.import_example(data='mnist')

print(X)
# Each row is an image that can be plotted after reshaping:
plt.imshow(X[0,:].reshape(8,8), cmap='binary')
# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  0., ...,  9.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  4.,  0.,  0.],
#        [ 0.,  0.,  6., ...,  6.,  0.,  0.]])
# 
# Preprocessing and feature extraction
results = cl.fit_transform(X)

# Lets examine the results.
print(results.keys())
# ['feat', 'xycoord', 'pathnames', 'filenames', 'labels']
# 
# feat      : Extracted features
# xycoord   : Coordinates of samples in the embedded space.
# filenames : Name of the files
# pathnames : Absolute location of the files
# labels    : Cluster labels in the same order as the input

# Get the unique images
unique_samples = cl.unique()
# 
print(unique_samples.keys())
# ['labels', 'idx', 'xycoord_center', 'pathnames']
# 
# Collect the unique images from the input
X[unique_samples['idx'],:]

```
##### Plot the unique images.

```python
cl.plot_unique()
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_unique.png" width="300" />
</p>


##### Scatter samples based on the embedded space.

```python
# The scatterplot that is coloured on the clusterlabels. The clusterlabels should match the unique labels.
# Cluster 1 contains digit 4
# Cluster 5 contains digit 2
# etc
# 
# No images in scatterplot
cl.scatter(zoom=None)

# Include images scatterplot
cl.scatter(zoom=4)

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig2_tsne.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig21_tsne.png" width="400" />
</p>



#### Plot the clustered images

```python
# Plot all images per cluster
cl.plot(cmap='binary')

# Plot the images in a specific cluster
cl.plot(cmap='binary', labels=[1,5])
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_cluster1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_cluster5.png" width="400" />
</p>


#### Dendrogram
```python
# The dendrogram is based on the high-dimensional feature space.
cl.dendrogram()

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_dendrogram.png" width="400" />
</p>


#### Make various other plots
```python
# Plot the explained variance
cl.pca.plot()
# Make scatter plot of PC1 vs PC2
cl.pca.scatter(legend=False, label=False)
# Plot the evaluation of the number of clusters
cl.clusteval.plot()
# Make silhouette plot
cl.clusteval.scatter(cl.results['xycoord'])
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_explained_var.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_clusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig1.png" width="600" />
</p>


### Example 2: Flower images.
In this example I will be using flower images for which the path locations are somewhere on disk.

```python
# Load library
from clustimage import Clustimage
# init
cl = Clustimage(method='pca')
# load example with flowers
pathnames = cl.import_example(data='flowers')
# The pathnames are stored in a list
print(pathnames[0:2])
# ['C:\\temp\\flower_images\\0001.png', 'C:\\temp\\flower_images\\0002.png']

# Preprocessing, feature extraction and clustering. Lets set a minimum of 1-
results = cl.fit_transform(pathnames)

# Lets first evaluate the number of detected clusters.
# This looks pretty good because there is a high distinction between the peak for 5 clusters and the number of clusters that subsequently follow.
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_sil_vs_nrclusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_silhouette.png" width="400" />
</p>


#### Scatter

```python
cl.scatter(dotsize=50, zoom=None)
cl.scatter(dotsize=50, zoom=0.5)
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter_imgs_mean.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter_imgs.png" width="300" />
</p>

#### Plot the clustered images

```python
# Plot unique images
cl.plot_unique()
cl.plot_unique(img_mean=False)

# Plot all images per cluster
cl.plot()

# Plot the images in a specific cluster
cl.plot(labels=3)
```

<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_unique.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_unique_mean.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_cluster3.png" width="400" />
</p>


```python
# Plot dendrogram
cl.dendrogram()
# Plot clustered images
cl.plot()

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_plot1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_plot2.png" width="400" />
</p>


#### Make prediction for unseen input image.
```python
# Find images that are significanly similar as the unseen input image. 
results_find = cl.find(path_to_imgs[0:2], alpha=0.05)
cl.plot_find()

# Map the unseen images in existing feature-space.
cl.scatter()
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_2.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_scatter.png" width="400" />
</p>


### Example 3: Cluster the faces on images.

```python
from clustimage import Clustimage
# Initialize with grayscale and extract HOG features.
cl = Clustimage(method='hog', grayscale=True)
# Load example with faces
pathnames = cl.import_example(data='faces')
# First we need to detect and extract the faces from the images
face_results = cl.extract_faces(pathnames)
# The detected faces are extracted and stored in face_resuls. We can now easily provide the pathnames of the faces that are stored in pathnames_face.
results = cl.fit_transform(face_results['pathnames_face'])

# Plot the evaluation of the number of clusters. As you can see, the maximum number of cluster evaluated is 24 can perhaps be too small.
cl.clusteval.plot()
# Lets increase the maximum number and clusters and run solely the clustering. Note that you do not need to fit_transform() anymore. You can only do the clustering now.
cl.cluster(max_clust=35)
# And plot again. As you can see, it keeps increasing which means that it may not found any local maximum anymore.
# When looking at the graph, we see a local maximum at 12 clusters. Lets go for that
cl.cluster(min_clust=12, max_clust=13)

# Lets plot the 12 unique clusters that contain the faces
cl.plot_unique()

# Scatter
cl.scatter(zoom=None)
cl.scatter(zoom=0.2)

# Make plot
cl.plot(show_hog=True, labels=[1,7])

# Plot faces
cl.plot_faces()
# Dendrogram depicts the clustering of the faces
cl.dendrogram()

```

<p align="center">

  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_sil_vs_nrclusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_set_max_clust.png" width="400" />

  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_unique.png" width="400" />

  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_scatter_no_img.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_scatter.png" width="400" />

  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_cluster0.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_cluster3.png" width="400" />

  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces1.png" width="400" />
</p>


### Maintainers

  * Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
  * https://github.com/erdogant/clustimage
  * Please cite in your publications if this is useful for your research (see citation).
  * All kinds of contributions are welcome!
  * If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)
  See [LICENSE](LICENSE) for details.

#### Other interesting stuff
* https://ourcodeworld.com/articles/read/1006/how-to-determine-whether-2-images-are-equal-or-not-with-the-perceptual-hash-in-python
* https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
* https://github.com/JohannesBuchner/imagehash
* https://ourcodeworld.com/articles/read/1006/how-to-determine-whether-2-images-are-equal-or-not-with-the-perceptual-hash-in-python
* https://stackoverflow.com/questions/64994057/python-image-hashing
* https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
