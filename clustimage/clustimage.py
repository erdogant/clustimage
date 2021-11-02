"""Clustimage"""
# --------------------------------------------------
# Name        : clustimage.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/clustimage
# Licence     : See licences
# --------------------------------------------------

from pca import pca
from distfit import distfit
import pandas as pd
import colourmap
import os
import logging
from urllib.parse import urlparse
import fnmatch
import zipfile
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [Clustimage]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Clustimage():
    """clustimage."""

    def __init__(self, method='pca', image_type='object', grayscale=False, dim=(128,128), verbose=20):
        """Initialize distfit with user-defined parameters."""
        # Reads the image as grayscale and results in a 2D-array. In case of RGB, no transparency channel is included.
        self.method = method
        self.image_type = image_type
        self.grayscale = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR # cv2.COLOR_BGR2RGB
        self.dim = dim
        set_logger(verbose=verbose)

    def predict(self, pathnames, metric='euclidean', k=1, alpha=0.05):
        """Import example dataset from github source."""
        if (k is None) and (alpha is None):
            raise Exception(logger.error('Nothing to collect! input parameter "k" and "alpha" can not be None at the same time.'))
        out = None
        # Read images and preprocessing. This is indepdent on the method type but should be in similar manner.
        X = self.fit(pathnames)

        # Predict according PCA method
        if self.method=='pca':
            Y, dist, feat = self.compute_distances_pca(X, metric=metric, alpha=alpha)
            out = self.collect_pca(X, Y, dist, k, alpha, feat, todf=True)

        # Store
        self.results['predict'] = out
        # Return
        return out

    def fit(self, pathnames):
        # Extact features
        logger.info("Reading images..")
        # Read and pre-proces the input images
        out = self.preprocessing(pathnames)
        logger.info("New feature matrix: %s", str(out['I'].shape))
        # Return
        return out

    def fit_transform(self, pathnames):
        """Import example dataset from github source."""
        # Read images and preprocessing
        X = self.fit(pathnames)
        # Extract features using method
        if self.method=='pca':
            self.model = pca(n_components=150)
            self.model.fit_transform(X['I'], row_labels=X['filenames'])
        # Store results
        self.results = {}
        self.results['feat'] = self.model.results['PC']
        self.results['pathnames'] = X['pathnames']
        self.results['filenames'] = X['filenames']
        # Return
        return self.results

    def preprocessing(self, pathnames):
        """Import example dataset from github source."""
        I, filenames = None, None
        if isinstance(pathnames, str):
            pathnames=[pathnames]
        if isinstance(pathnames, list):
            filenames = list(map(basename, pathnames))
            I = list(map(lambda x: img_read_pipeline(x, grayscale=self.grayscale, dim=self.dim, flatten=True), pathnames))
            I = np.vstack(I)
        # else:
        #     # Use the image values directly
        #     labels = np.arange(0, X.shape[0]).astype(str)
        #     X = list(map(lambda x: img_read(x, grayscale=self.grayscale), X))
        #     X = list(map(lambda x: img_resize(x, dim=self.dim), X))
        #     X = list(map(img_flatten, X))
        #     X = np.vstack(X)
        out = {}
        out['I'] = I
        out['pathnames'] = pathnames
        out['filenames'] = filenames
        return out

    # Compute distances and probabilities after transforming the data using PCA.
    def compute_distances_pca(self, X, metric, alpha):
        dist = None
        # Transform new "unseen" data. Note that these datapoints are not really unseen as they are readily fitted above.
        PCnew = self.model.transform(X['I'], row_labels=X['filenames'])
        # Compute distance to all samples
        Y = distance.cdist(self.results['feat'], PCnew, metric=metric)
        Ytot = distance.cdist(self.results['feat'], self.results['feat'], metric=metric)
        # Fit distribution to emperical data and compute probability of the distances of interest
        if alpha is not None:
            dist = distfit(bound='down', multtest=None)
            dist.fit_transform(Ytot)
            # model.plot()

        # Sanity check
        if len(X['filenames'])!=Y.shape[1]: raise Exception(logger.error('Number of input files does not match number of computed distances.'))
        # Return
        return Y, dist, PCnew
        
    # Collect data
    def collect_pca(self, X, Y, dist, k, alpha, feat, todf=True):
        # Collect the samples that are closest in according the metric
        filenames = X['filenames']
        out = {}
        out['feat'] = feat

        # Collect nearest neighor and sample with highes probability per input sample
        for i, filename in enumerate(filenames):
            logger.info('Predict: %s', filename)
            store_key = {}
            idx_dist, idx_k = None, None
            # Collect bes samples based on k-nearest neighbor
            if k is not None:
                idx_k = np.argsort(Y[:,i])[0:k]
            # Collect samples based on probability
            if alpha is not None:
                dist_results = dist.predict(Y[:,i], verbose=0)
                idx_dist = np.where(dist_results['y_proba']<=alpha)[0]
            else:
                # If alha is not used, set all to nan
                dist_results={}
                dist_results['y_proba'] = np.array([np.nan]*Y.shape[0])

            # Combine the unique k-nearest samples and probabilities.
            idx = unique_no_sort(np.append(idx_dist, idx_k))

            # Store in dict
            store_key = {**store_key, 'x_path': X['pathnames'][i], 'y_idx': idx, 'distance': Y[idx, i], 'y_proba': dist_results['y_proba'][idx], 'y_label': np.array(self.results['filenames'])[idx].tolist(), 'y_path': np.array(self.results['pathnames'])[idx].tolist()}
            if todf: pd.DataFrame(store_key)
            out[filename] = store_key

        # Return
        return out

    def plot(self, legend=False):
        fig, ax = self.model.plot()
        # Show the first eigenvectors
        # self.model.scatter3d(legend=legend)

        # Scatter all points
        fig, ax = self.model.scatter(y=1, legend=legend, label=False)

        # Scatter the predicted cases
        if self.results.get('predict', None) is not None:
            # Create unique colors
            colours = colourmap.fromlist(self.results['predict']['feat'].index)[1]
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    x,y,z = self.results['predict']['feat'].loc[key][0:3]
                    idx = self.results['predict'][key]['y_idx']
                    # Scatter
                    ax.scatter(x, y, color=colours[key], edgecolors=[0,0,0])
                    ax.text(x,y, key, color=colours[key])
                    ax.scatter(self.results['feat'].iloc[idx].iloc[:,0], self.results['feat'].iloc[idx].iloc[:,1], edgecolors=[0,0,0])

            # Plot the images that are similar to each other.
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    input_img = self.results['predict'][key]['x_path']
                    predict_img = self.results['predict'][key]['y_path']

                    # Input label
                    # input_img = img[np.isin(labels, key)]
                    if isinstance(input_img, str): input_img=[input_img]

                    # Predicted label
                    # predict_img = img[np.isin(labels, self.results['predict'][key]['label'])]
                    # Input images
                    I_input = list(map(lambda x: img_read_pipeline(x, grayscale=self.grayscale, dim=self.dim, flatten=False), input_img))
                    I_predict = list(map(lambda x: img_read_pipeline(x, grayscale=self.grayscale, dim=self.dim, flatten=False), predict_img))

                    fig, axes = plt.subplots(len(I_predict)+1,1,sharex=True,sharey=True,figsize=(8,10))
                    axes[0].set_title('Input image')
                    axes[0].imshow(I_input[0])
                    for i, I in enumerate(I_predict):
                        axes[i+1].set_title('Predicted: %s' %(i+1))
                        axes[i+1].imshow(I)


    def import_example(self, data='flower', url=None):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            'flower'

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url)


# %% Unique without sort
def unique_no_sort(x):
    x = x[x!=None]
    indexes = np.unique(x, return_index=True)[1]
    return [x[index] for index in sorted(indexes)]

# %% Resize image
def basename(label):
    return os.path.basename(label)


# %% Resize image
def img_flatten(img):
    return img.flatten()


# %% Resize image
def img_resize(img, dim=(128, 128)):
    if dim is not None:
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


# %% Read image
def img_read(filepath, grayscale=1):
    img=None
    if os.path.isfile(filepath):
        # Read the image
        img = cv2.imread(filepath, grayscale)
    else:
        logger.warning('File does not exists: %s', filepath)

    return img


# %% Read image
def img_read_pipeline(filepath, grayscale=1, dim=(128, 128), flatten=True):
    # Read the image
    img = img_read(filepath, grayscale=grayscale)
    # Resize the image
    img = img_resize(img, dim=dim)
    # Flatten the image
    if flatten: img = img_flatten(img)
    return img

# %%
def set_logger(verbose=20):
    logger.setLevel(verbose)


# %% Import example dataset from github.
def import_example(data='flower', url=None):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'flower'
    url : str
        url link to to dataset.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='flower':
            url='https://erdogant.github.io/datasets/flower_images.zip'
    else:
        logger.warning('Lets try your dataset from url: %s.', url)

    if url is None:
        logger.info('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = os.path.basename(urlparse(url).path)
    path_to_data = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(path_to_data):
        logger.info('Downloading [%s] dataset from github source..', data)
        wget(url, path_to_data)

    # Unzip
    dirpath = _unzip(path_to_data)
    # Import local dataset
    image_files = get_images_recursive(dirpath)
    # Return
    return image_files


# %% Recursively list files from directory
def get_images_recursive(filepath, ext=['png','tiff','jpg']):
    logger.info('Retrieve files recursively from path: [%s]', filepath)
    getfiles = []
    for iext in ext:
        for root, _, filenames in os.walk(filepath):
            for filename in fnmatch.filter(filenames, '*.'+iext):
                getfiles.append(os.path.join(root, filename))
    return getfiles


# %% unzip
def _unzip(path_to_zip):
    getpath = None
    if path_to_zip[-4:]=='.zip':
        if not os.path.isdir(path_to_zip):
            logger.info('Extracting files..')
            pathname, _ = os.path.split(path_to_zip)
            # Unzip
            zip_ref = zipfile.ZipFile(path_to_zip, 'r')
            zip_ref.extractall(pathname)
            zip_ref.close()
            getpath = path_to_zip.replace('.zip', '')
            if not os.path.isdir(getpath):
                logger.error('Extraction failed.')
                getpath = None
    else:
        logger.warning('Input is not a zip file: [%s]', path_to_zip)
    # Return
    return getpath


# %% Download files from github source
def wget(url, writepath):
    r = requests.get(url, stream=True)
    with open(writepath, "wb") as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)


# %% Main
# if __name__ == "__main__":
#     import clustimage as clustimage
#     df = clustimage.import_example()
#     out = clustimage.fit(df)
#     fig,ax = clustimage.plot(out)
