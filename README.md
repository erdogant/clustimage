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
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

* clustimage is a python package for unsupervised clustering of images.


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

#### Simple example using data-array as an input.
```python
# Load library
import matplotlib.pyplot as plt
from clustimage import Clustimage
# init
cl = Clustimage()
# Load example digit data
X = cl.import_example(data='digits')

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

# Scatter
cl.scatter()
# Plot dendrogram
cl.dendrogram()
# Plot the clustered images
cl.plot(cmap='binary')

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig1.png" width="600" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_explained_var.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_pca.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_fig2_tsne.png" width="600" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_clusters.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_dendrogram.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_cluster1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/digits_cluster4.png" width="400" />
</p>


#### images with flowers to cluster.
```python
# Load library
from clustimage import Clustimage
# init
cl = Clustimage(method='pca', embedding='tsne')
# load example with flowers
path_to_imgs = cl.import_example(data='flowers')
# Preprocessing and feature extraction
results = cl.fit_transform(path_to_imgs, min_clust=10)
# Scatter
cl.scatter(dot_size=50)
# Plot dendrogram
cl.dendrogram()
# Plot clustered images
cl.plot(ncols=5)

# Predict
results_predict = cl.predict(path_to_imgs[0:5], k=None, alpha=0.05)
cl.plot_predict()
cl.scatter()

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_plot1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flowers_plot2.png" width="400" />
</p>


#### Make prediction with unknown input image.
```python
# Predict
results_predict = cl.predict(path_to_imgs[0:5], alpha=0.05)
cl.plot_predict()
cl.scatter()
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/scatter_predict.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/flower_predict_example.png" width="400" />
</p>


#### Make prediction with unknown input image.
```python
from clustimage import Clustimage
# Init
cl = Clustimage(method='pca', grayscale=True, params_pca={'n_components':14})
# Load example with faces
pathnames = cl.import_example(data='faces')
# Detect faces
face_results = cl.detect_faces(pathnames)
# Cluster
results = cl.fit_transform(face_results['facepath'])

# Plot faces
cl.plot_faces()
# Plot dendrogram
cl.dendrogram()

# Make various other plots
cl.scatter()
# Cluster
labx = cl.cluster()
# Make plot
cl.plot(ncols=2, show_hog=True)
```

<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces1.png" width="400" />
  <img src="https://github.com/erdogant/clustimage/blob/main/docs/figs/faces_dendrogram.png" width="400" />
</p>


#### References
* https://github.com/erdogant/clustimage

#### Citation
Please cite in your publications if this is useful for your research (see citation).
   
### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

### Licence
See [LICENSE](LICENSE) for details.
