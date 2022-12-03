# clustimage

[![Python](https://img.shields.io/pypi/pyversions/clustimage)](https://img.shields.io/pypi/pyversions/clustimage)
[![Pypi](https://img.shields.io/pypi/v/clustimage)](https://pypi.org/project/clustimage/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/clustimage/)
[![LOC](https://sloc.xyz/github/erdogant/clustimage/?category=code)](https://github.com/erdogant/clustimage/)
[![Downloads](https://static.pepy.tech/personalized-badge/clustimage?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/clustimage)
[![Downloads](https://static.pepy.tech/personalized-badge/clustimage?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/clustimage)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clustimage/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/clustimage.svg)](https://github.com/erdogant/clustimage/network)
[![Issues](https://img.shields.io/github/issues/erdogant/clustimage.svg)](https://github.com/erdogant/clustimage/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/423822054.svg)](https://zenodo.org/badge/latestdoi/423822054)
[![Medium](https://img.shields.io/badge/Medium-Blog-blue)](https://erdogant.github.io/clustimage/pages/html/Documentation.html#)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg?logo=github%20sponsors)](https://erdogant.github.io/clustimage/pages/html/Documentation.html#colab-notebook)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/clustimage/pages/html/Documentation.html#)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->


The aim of ``clustimage`` is to detect natural groups or clusters of images. It works using a multi-step proces of carefully pre-processing the images, extracting the features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters can be determined using well known methods suchs as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
With ``clustimage`` we aim to determine the most robust clustering by efficiently searching across the parameter and evaluation the clusters.
Besides clustering of images, the ``clustimage`` model can also be used to find the most similar images for a new unseen sample.

A schematic overview is as following:

<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/schematic_overview.png" width="1000" />
</p>

``clustimage`` overcomes the following challenges: 

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


# 
**⭐️ Star this repo if you like it ⭐️**
#

### Blogs

* Read the [blog](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128) to get a structured overview how to cluster images.

# 

### [Documentation pages](https://erdogant.github.io/clustimage/)

On the [documentation pages](https://erdogant.github.io/clustimage/) you can find detailed information about the working of the ``clustimage`` with many examples. 

# 


### Installation

##### It is advisable to create a new environment (e.g. with Conda). 
```bash
conda create -n env_clustimage python=3.8
conda activate env_clustimage
```

##### Install bnlearn from PyPI
```bash
pip install clustimage            # new install
pip install -U clustimage         # update to latest version
```

##### Directly install from github source
```bash
pip install git+https://github.com/erdogant/clustimage
```  

##### Import clustimage package

```python
from clustimage import clustimage
```

<hr>

### Examples

The results obtained from the clustimgage library is a dictionary containing the following keys:

    * img       : image vector of the preprocessed images
    * feat      : Features extracted for the images
    * xycoord   : X and Y coordinates from the embedding
    * pathnames : Absolute path location to the image file
    * filenames : File names of the image file
    * labels    : Cluster labels


### Examples Mnist dataset:

##### [Example: Clustering mnist dataset](https://erdogant.github.io/clustimage/pages/html/Examples.html#)

In this example we will be using a flattened grayscale image array loaded from sklearn. The unique detected clusters are the following:

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#scatter-plot">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig2_tsne.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig21_tsne.png" width="400" />
  </a>
</p>

**Click on the underneath scatterplot to zoom-in and see ALL the images in the scatterplot**

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/scatter_mnist_all.png" width="400" />
</p>


#

##### [Example: Plot the explained variance](https://erdogant.github.io/clustimage/pages/html/Examples.html#cluster-evaluation)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#cluster-evaluation">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_explained_var.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_clusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig1.png" width="600" />
  </a>
</p>

#

##### [Example: Plot the unique images](https://erdogant.github.io/clustimage/pages/html/Examples.html#detect-unique-images)


<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#detect-unique-images">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_unique.png" width="300" />
  </a>
</p>


#


##### [Example: Plot the dendrogram](https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_dendrogram.png" width="400" />
  </a>
</p>



<hr> 


### Examples Flower dataset:

##### [Example: cluster the flower dataset](https://erdogant.github.io/clustimage/pages/html/Examples.html#id5)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#id5">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_sil_vs_nrclusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_silhouette.png" width="400" />
  </a>
</p>


##### [Example: Make scatterplot with clusterlabels](https://erdogant.github.io/clustimage/pages/html/Examples.html#id7)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#id7">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter_imgs_mean.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter_imgs.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_scatter_all.png" width="300" />
  </a>
</p>


##### [Example: Plot the unique images per cluster](https://erdogant.github.io/clustimage/pages/html/Examples.html#id6)

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_unique.png" width="400" />
</p>

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_unique_mean.png" width="400" />
</p>


##### [Example: Plot the images in a particular cluster](https://erdogant.github.io/clustimage/pages/html/Examples.html#id8)

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_cluster3.png" width="400" />
</p>




##### [Example: Make prediction for unseen input image](https://erdogant.github.io/clustimage/pages/html/Examples.html#predict-unseen-sample)

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_1.png" width="600" />
</p>
<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_2.png" width="600" />
</p>
<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_scatter.png" width="600" />
</p>


<hr> 


#### [Example: Clustering of faces on images](https://erdogant.github.io/clustimage/pages/html/Examples.html#clustering-of-faces)


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

<hr>

#### [Example: Break up the steps](https://erdogant.github.io/clustimage/pages/html/Examples.html#breaking-up-the-steps)

<hr>

#### [Example: Extract images belonging to clusters](https://erdogant.github.io/clustimage/pages/html/Examples.html#extract-images-belonging-to-clusters)

<hr>


### Support

	This project needs some love! ❤️ You can help in various ways.

	* Become a Sponsor!
	* Star this repo at the github page.
	* Other contributions can be in the form of feature requests, idea discussions, reporting bugs, opening pull requests.
	* Read more why becoming an sponsor is important on the Sponsor Github Page.
	
	Cheers Mate.
