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

* clustimage is Python package

### Contents
- [Installation](#-installation)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install clustimage from PyPI (recommended). clustimage is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```bash
conda create -n env_clustimage python=3.7
conda activate env_clustimage
```

```bash
pip install clustimage            # normal install
pip install --upgrade clustimage # or update if needed
```

* Alternatively, you can install from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/clustimage.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/clustimage#egg=master
pip install git+https://github.com/erdogant/clustimage

# By cloning
git clone https://github.com/erdogant/clustimage.git
cd clustimage
pip install -U .
```  

#### Import clustimage package
```python
import clustimage as clustimage
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/clustimage/data/example_data.csv')
model = clustimage.fit(df)
G = clustimage.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/clustimage/blob/master/docs/figs/fig1.png" width="600" />
  
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
