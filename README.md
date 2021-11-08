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

#### When input is data-array.
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

# Preprocessing and feature extraction
results = cl.fit_transform(X)

# Scatter
cl.scatter()

```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/master/docs/figs/digits_fig1.png" width="600" />
  <img src="https://github.com/erdogant/clustimage/blob/master/docs/figs/digits_explained_var.png" width="600" />
  <img src="https://github.com/erdogant/clustimage/blob/master/docs/figs/digits_pca.png" width="600" />
  <img src="https://github.com/erdogant/clustimage/blob/master/docs/figs/digits_fig2_tsne.png" width="600" />
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
