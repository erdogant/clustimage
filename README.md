# clustimage

[![Python](https://img.shields.io/pypi/pyversions/clustimage)](https://img.shields.io/pypi/pyversions/clustimage)
[![PyPI Version](https://img.shields.io/pypi/v/clustimage)](https://pypi.org/project/clustimage/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clustimage/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/clustimage.svg)](https://github.com/erdogant/clustimage/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/clustimage.svg)](https://github.com/erdogant/clustimage/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/clustimage/month)](https://pepy.tech/project/clustimage/month)
[![Downloads](https://pepy.tech/badge/clustimage)](https://pepy.tech/project/clustimage)
[![DOI](https://zenodo.org/badge/423822054.svg)](https://zenodo.org/badge/latestdoi/423822054)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erdogant/clustimage/blob/master/notebooks/clustimage.ipynb)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/clustimage/)
[![Medium](https://img.shields.io/badge/Medium-Blog-blue)](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128)
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


##### [Example: Plot the images per cluster](https://erdogant.github.io/clustimage/pages/html/Examples.html#plot-images-detected-in-a-cluster)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#plot-images-detected-in-a-cluster">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig2_tsne.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig21_tsne.png" width="400" />
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

##### [Example: cluster the flower dataset](https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_sil_vs_nrclusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_silhouette.png" width="400" />
  </a>
</p>


##### [Example: Make scatterplot with clusterlabels](https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram)

<p align="left">
  <a href="https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter_imgs_mean.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_scatter_imgs.png" width="300" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_scatter_all.png" width="300" />
  </a>
</p>


##### [Example: Plot the unique images per cluster](https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram)

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_unique.png" width="400" />
</p>

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_unique_mean.png" width="400" />
</p>


##### [Example: Plot the images in a particular cluster](https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram)

<p align="left">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_cluster3.png" width="400" />
</p>




##### [Example: Make prediction for unseen input image](https://erdogant.github.io/clustimage/pages/html/Examples.html#dendrogram)

<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_2.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_predict_scatter.png" width="400" />
</p>



### Example 3: Cluster the faces on images.

```python
from clustimage import Clustimage
# Initialize with PCA
cl = Clustimage(method='pca', grayscale=True)
# Load example with faces
X = cl.import_example(data='faces')
# Initialize and run
results = cl.fit_transform(X)

# In case you need to extract the faces from the images
# face_results = cl.extract_faces(pathnames)
# The detected faces are extracted and stored in face_resuls. We can now easily provide the pathnames of the faces that are stored in pathnames_face.
# results = cl.fit_transform(face_results['pathnames_face'])

# Plot the evaluation of the number of clusters. As you can see, the maximum number of cluster evaluated is 24 can perhaps be too small.
cl.clusteval.plot()
# Lets increase the maximum number and clusters and run solely the clustering. Note that you do not need to fit_transform() anymore. You can only do the clustering now.
cl.cluster(max_clust=35)
# And plot again. As you can see, it keeps increasing which means that it may not found any local maximum anymore.
# When looking at the graph, we see a local maximum at 12 clusters. Lets go for that
cl.cluster(min_clust=4, max_clust=20)

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


### Example 4: Break up the steps

Instead of using the all-in-one functionality: fit_transform(), it is also possible to break-up the steps.

```python

from clustimage import Clustimage

# Initialize
cl = Clustimage(method='pca')

# Import data
Xraw = cl.import_example(data='flowers')
Xraw = cl.import_example(data='mnist')
Xraw = cl.import_example(data='faces')

# Check whether in is dir, list of files or array-like
X = cl.import_data(Xraw)

# Extract features using method
Xfeat = cl.extract_feat(X)

# Embedding using tSNE
xycoord = cl.embedding(Xfeat)

# Cluster
labels = cl.cluster()

# Return
results = cl.results

# Or all in one run
# results = cl.fit_transform(X)

# Plots
cl.clusteval.plot()
cl.scatter()
cl.plot_unique()
cl.plot()
cl.dendrogram()

# Find
results_find = cl.find(Xraw[0], k=0, alpha=0.05)
cl.plot_find()
```
### Example: Extract images belonging to clusters

The results obtained from the cl.fit_transform() or cl.cluster() is a dictionary containing the following keys:

    * img       : image vector of the preprocessed images
    * feat      : Features extracted for the images
    * xycoord   : X and Y coordinates from the embedding
    * pathnames : Absolute path location to the image file
    * filenames : File names of the image file
    * labels    : Cluster labels

```python

# Import library
from clustimage import Clustimage
# Initialize
cl = Clustimage(method='pca')
# Import data
pathnames = cl.import_example(data='flowers')
# Cluster flowers
results = cl.fit_transform(pathnames)

# All results are stored in a dict:
print(cl.results.keys())
# Which is the same as:
print(results.keys())

dict_keys(['img', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames'])

# Extracting images that belong to cluster label=0:
Iloc = cl.results['labels']==0
cl.results['pathnames'][Iloc]

# Extracting xy-coordinates for the scatterplot for cluster 0:
import matplotlib.pyplot as plt
xycoord = cl.results['xycoord'][Iloc]
plt.scatter(xycoord[:,0], xycoord[:,1])

# Plot the images for cluster 0:
# Images in cluster 0
imgs = np.where(cl.results['img'][Iloc])[0]
# Make sure you get the right dimension
dim = cl.get_dim(cl.results['img'][Iloc][0,:])
# Plot
for img in imgs:
  plt.figure()
  plt.imshow(img.reshape(dim))
  plt.title()

```

### Maintainers

  * Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
  * https://github.com/erdogant/clustimage
  * Please cite in your publications if this is useful for your research (see citation).
  * All kinds of contributions are welcome!
  * If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)
  See [LICENSE](LICENSE) for details.

#### Other interesting stuff
* https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
