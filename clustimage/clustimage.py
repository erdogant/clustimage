"""Python package clustimage is for unsupervised clustering of images."""
# --------------------------------------------------
# Name        : clustimage.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/clustimage
# Licence     : See licences
# --------------------------------------------------

from pca import pca
from distfit import distfit
from clusteval import clusteval
from scatterd import scatterd
import pypickle as pypickle
import pandas as pd
import colourmap
from tqdm import tqdm
from sklearn.manifold import TSNE
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
from skimage.feature import hog
from skimage import exposure
import tempfile
import uuid
import shutil
import random

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
formatter = logging.Formatter('[clustimage] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Clustimage():
    """Clustering of images.

    Description
    -----------
    Clustering input images after following steps of pre-processing, feature-extracting, feature-embedding and cluster-evaluation.
    Taking all these steps requires setting various input parameters. Not all input parameters can be changed across the different steps in clustimage.
    Some parameters are choosen based on best practice, some parameters are optimized, while others are set as a constant.
    The following 4 steps are taken:

    Step 1. Pre-processing:
        Images are imported with specific extention (['png','tiff','jpg']), 
        Each input image can then be grayscaled. Setting the grayscale parameter to True can be especially usefull when clustering faces.
        Final step in pre-processing is resizing all images in the same dimension such as (128,128). Note that if an array-like dataset [Samples x Features] is given as input, setting these dimensions are required to restore the image in case of plotting.
    Step 2. Feature-extraction:
        Features are extracted from the images using Principal component analysis (PCA), Histogram of Oriented Gradients (HOG) or the raw values are used.
    Step 3. Embedding:
        The feature-space non-lineair transformed using t-SNE and the coordinates are stored. The embedding is only used for visualization purposes.
    Step 4. Cluster evaluation:
        The feature-space is used as an input in the cluster-evaluation method. The cluster evaluation method determines the optimal number of clusters and return the cluster labels.
    Step 5: Done.
        The results are stored in the object and returned by the model. Various different (scatter) plots can be made to evaluate the results.

    Parameters
    ----------
    method : str, (default: 'pca')
        Method to be usd to extract features from images.
            * 'pca' : PCA feature extraction
            * 'hog' : hog features extraced
            * None : No feature extraction
    embedding : str, (default: 'tsne')
        Perform embedding on the extracted features. The xycoordinates are used for plotting purposes.
            * 'tsne' or  None
    grayscale : Bool, (default: False)
        Colorscaling the image to gray. This can be usefull when clustering e.g., faces.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    dirpath : str, (default: None)
        Directory to write images.
    ext : list, (default: ['png','tiff','jpg'])
        Images with the file extentions are used.
    params_pca : dict, default: {'n_components':50, 'detect_outliers':None}
        Parameters to initialize the pca model.
    params_hog : dict, default: {'orientations':9, 'pixels_per_cell':(16,16), 'cells_per_block':(1,1)}
        Parameters to extract hog features.
    verbose : int, (default: 20)
        Print progress to screen. The default is 3.
        60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

    Returns
    -------
    Object.
    model : dict
        dict containing keys with results.
        feat : array-like.
            Features extracted from the input-images
        xycoord : array-like.
            x,y coordinates after embedding or alternatively the first 2 features.
        pathnames : list of str.
            Full path to images that are used in the model.
        filenames : list of str.
            Filename of the input images.
        labx : list.
            Cluster labels

    Example
    -------
    >>> from clustimage import Clustimage
    >>>
    >>> # Init with default settings
    >>> cl = Clustimage()
    >>> # load example with flowers
    >>> path_to_imgs = cl.import_example(data='flowers')
    >>> # Detect cluster
    >>> results = cl.fit_transform(path_to_imgs, min_clust=10)
    >>>
    >>> # Plot dendrogram
    >>> cl.dendrogram()
    >>> # Scatter
    >>> cl.scatter(dotsize=50)
    >>> # Plot clustered images
    >>> cl.plot()
    >>>
    >>> # Make prediction
    >>> results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
    >>> cl.plot_find()
    >>> cl.scatter()

    """
    def __init__(self, method='pca', embedding='tsne', grayscale=False, dim=(128,128), dim_face=(64,64), dirpath=None, store_to_disk=False, ext=['png','tiff','jpg'], params_pca={'n_components':50, 'detect_outliers':None}, params_hog={'orientations':9, 'pixels_per_cell':(16,16), 'cells_per_block':(1,1)}, verbose=20):
        """Initialize clustimage with user-defined parameters."""

        if not np.any(np.isin(method, [None, 'pca','hog'])): raise Exception(logger.error('method: "%s" is unknown', method))
        if dirpath is None: dirpath = tempfile.mkdtemp()
        if not os.path.isdir(dirpath): raise Exception(logger.error('[%s] does not exists.', dirpath))

        # Find path of xml file containing haarcascade file and load in the cascade classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        # self.params = {}
        # self.params['face_cascade'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # self.params['eye_cascade'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.method = method
        self.embedding = embedding
        self.grayscale = grayscale
        self.cv2_imread_colorscale = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        self.dim = dim
        self.dim_face = dim_face
        self.params_pca = params_pca
        self.dirpath = dirpath
        self.tempdir = tempfile.mkdtemp()
        self.ext = ext
        self.store_to_disk = store_to_disk
        set_logger(verbose=verbose)

    def fit_transform(self, X, cluster='agglomerative', method='silhouette', metric='euclidean', linkage='ward', min_clust=3, max_clust=25, cluster_space='high'):
        """Group samples into clusters that are similar in their feature space.

        Parameters
        ----------
        X : str, list or array-like.
            The input can be:
                * "c://temp//" : Path to directory with images
                * ['c://temp//image1.png', 'c://image2.png', ...] : List of exact pathnames.
                * [[.., ..], [.., ..], ...] : Array-like matrix in the form of [sampels x features]
        cluster : str, (default: 'agglomerative')
            Type of clustering.
                * 'agglomerative'
                * 'kmeans'
                * 'dbscan'
                * 'hdbscan'
        method : str, (default: 'silhouette')
            Method type for cluster validation.
                * 'silhouette'
                * 'dbindex'
                * 'derivative'
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
                * 'ward'
                * 'single'
                * 'complete'
                * 'average'
                * 'weighted'
                * 'centroid'
                * 'median'
        min_clust : int, (default: 2)
            Number of clusters that is evaluated greater or equals to min_clust.
        max_clust : int, (default: 25)
            Number of clusters that is evaluated smaller or equals to max_clust.
        cluster_space: str, (default: 'high')
            Selection of the features that are used for clustering. This can either be on high or low feature space.
                * 'high' : Original feature space.
                * 'low' : Input are the xycoordinates that are determined by "embedding". Thus either tSNE coordinates or the first two PCs or HOGH features.

        Returns
        -------
        Object.
        model : dict
            dict containing keys with results.
            feat : array-like.
                Features extracted from the input-images
            xycoord : array-like.
                x,y coordinates after embedding or alternatively the first 2 features.
            pathnames : list of str.
                Full path to images that are used in the model.
            filenames : list of str.
                Filename of the input images.
            labx : list.
                Cluster labels

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage(method='pca', grayscale=True, params_pca={'n_components':10})
        >>> # load example with faces
        >>> pathnames = cl.import_example(data='faces')
        >>> # Detect faces
        >>> face_results = cl.detect_faces(pathnames)
        >>> # Cluster extracted faces
        >>> results = cl.fit_transform(face_results['facepath'])
        >>>
        >>> # Plot dendrogram
        >>> cl.dendrogram()
        >>> # Scatter
        >>> cl.scatter(dotsize=100)
        >>> # Plot clustered images
        >>> cl.plot(ncols=2)
        >>> # Plot facces
        >>> cl.plot_faces()
        >>>
        >>> # Make prediction
        >>> results_find = cl.find(face_results['facepath'][2][0], k=None, alpha=0.05)
        >>> cl.plot_find()
        >>> cl.scatter()

        """
        # Clean readily fitted models to ensure correct results
        self._clean()
        # Check whether in is dir, list of files or array-like
        X = self._import_data(X)
        # Extract features using method
        raw, X = self._extract_feat(X)
        # Embedding using tSNE
        xycoord = self._compute_embedding(X)
        # Store results
        self.results = {}
        self.results['feat'] = X
        self.results['xycoord'] = xycoord
        self.results['pathnames'] = raw['pathnames']
        self.results['filenames'] = raw['filenames']
        # Cluster
        self.cluster(cluster=cluster, method=method, cluster_space=cluster_space, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust)
        # Return
        return self.results

    def cluster(self, cluster='agglomerative', method='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, cluster_space='high'):
        """Detection of the optimal number of clusters given the input set of features.
        
        Description
        -----------
        This function is build on clusteval, which is a python package that provides various methods for unsupervised cluster validation.

        Parameters
        ----------
        cluster_space : str, (default: 'high')
            Selection of the features that are used for clustering. This can either be on high or low feature space.
                * 'high' : Original feature space.
                * 'low' : Input are the xycoordinates that are determined by "embedding". Thus either tSNE coordinates or the first two PCs or HOGH features.
        cluster : str, (default: 'agglomerative')
            Type of clustering.
                * 'agglomerative'
                * 'kmeans'
                * 'dbscan'
                * 'hdbscan'
        method : str, (default: 'silhouette')
            Method type for cluster validation.
                * 'silhouette'
                * 'dbindex'
                * 'derivative'
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
                * 'ward'
                * 'single'
                * 'complete'
                * 'average'
                * 'weighted'
                * 'centroid'
                * 'median'
        min_clust : int, (default: 2)
            Number of clusters that is evaluated greater or equals to min_clust.
        max_clust : int, (default: 25)
            Number of clusters that is evaluated smaller or equals to max_clust.
        savemem : bool, (default: False)
            Save memmory when working with large datasets. Note that htis option only in case of KMeans.

        Returns
        -------
        array-like
            .results['labx'] : Cluster labels.
            .clusteval : model parameters for cluster-evaluation and plotting.

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage(method='pca')
        >>> # load example with digits
        >>> X = cl.import_example(data='digits')
        >>> # Find clusters
        >>> results = cl.fit_transform(X)
        >>> # Scatter
        >>> cl.scatter(dotsize=25)
        >>>
        >>> # Change the clustering method, metric, minimum expected nr. of clusters etc.
        >>> labx = cl.cluster(cluster='agglomerative', method='dbindex', metric='euclidean', linkage='ward', min_clust=2, max_clust=25)
        >>>
        >>> # Scatter
        >>> cl.scatter(dotsize=25)
        >>>
        >>> # Plot clustered images
        >>> cl.plot(cmap='binary')
        >>> # Plot dendrogram
        >>> cl.dendrogram()

        """
        if self.results.get('feat', None) is None: raise Exception(logger.error('First run the "fit_transform(pathnames)" function.'))
        self.cluster_space = cluster_space
        # Init
        ce = clusteval(cluster=cluster, method=method, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, verbose=3)
        # Fit
        if cluster_space=='low':
            feat = self.results['xycoord']
            logger.info('Cluster evaluation using the [%s] feature space of the [%s] coordinates.', cluster_space, self.embedding)
        else:
            feat = self.results['feat']
            logger.info('Cluster evaluation using the [%s] feature space of the [%s] features.', cluster_space, self.method)
        # Fit model
        ce.fit(feat)
        # Store
        logger.info('Updating cluster-labels and cluster-model based on the %s feature-space.', str(feat.shape))
        self.results['labx'] = ce.results['labx']
        self.clusteval = ce
        # Return
        return ce.results['labx']

    def find(self, X, metric='euclidean', k=1, alpha=0.05):
        """Find images that are similar to that of the input image.

        Description
        -----------
        Computes K-nearest neighbour and P-values for pathnames [y] based on the fitted distribution from X.
        The empirical distribution of X is used to estimate the loc/scale/arg parameters for a theoretical distribution.
        For each image, the probability is computed for input variables y, and returns the images that are <= alpha
        If both k and alpha is specified, the union is taken.

        Parameters
        ----------
        pathnames : list of str.
            Full path to images that are used in the model.
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        k : int, (default: 1)
            The k-nearest neighbour.
        alpha : float, default: 0.05
            Significance alpha.

        Returns
        -------
        dict.
            Images are returned that are either k-nearest neighbour or significant.

        """
        if (k is None) and (alpha is None):
            raise Exception(logger.error('Nothing to collect! input parameter "k" and "alpha" can not be None at the same time.'))
        out = None

        # Check whether in is dir, list of files or array-like
        X = self._import_data(X)

        # Read images and preprocessing. This is indepdent on the method type but should be in similar manner.
        # X = self.preprocessing(pathnames, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=True)

        # Predict according PCA method
        if self.method=='pca':
            Y, feat = self._compute_distances_pca(X, metric=metric, alpha=alpha)
            out = self._collect_pca(X, Y, k, alpha, feat, todf=True)
        else:
            logger.warning('Nothing to Find. Prediction requires initialization with method="pca".')

        # Store
        self.results['predict'] = out
        # Return
        return self.results['predict']

    def detect_faces(self, pathnames):
        """Detect and extract faces from images.

        Parameters
        ----------
        pathnames : list of str.
            Full path to images that are used in the model.

        Returns
        -------
        Object.
        model : dict
            dict containing keys with results.
            pathnames : list of str.
                Full path to images that are used in the model.
            filenames : list of str.
                Filename of the input images.
            facepath : list of str.
                Filename of the extracted faces that are stored to disk.
            img : array-like.
                NxMxC for which N are the Samples, M the features and C the number of channels.
            coord_faces : array-like.
                list of lists containing coordinates fo the faces in the original image.
            coord_eyes : array-like.
                list of lists containing coordinates fo the eyes in the extracted (img and facepath) image.

        Example
        -------

        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage(method='pca', grayscale=True, params_pca={'n_components':10})
        >>> # load example with faces
        >>> pathnames = cl.import_example(data='faces')
        >>> # Detect faces
        >>> face_results = cl.detect_faces(pathnames)
        >>> # Cluster extracted faces
        >>> results = cl.fit_transform(face_results['facepath'])
        >>>
        >>> # Plot dendrogram
        >>> cl.dendrogram()
        >>> # Scatter
        >>> cl.scatter(dotsize=100)
        >>> # Plot clustered images
        >>> cl.plot(ncols=2)
        >>> # Plot facces
        >>> cl.plot_faces()
        >>>
        >>> # Find image
        >>> results_find = cl.find(face_results['facepath'][2], k=None, alpha=0.05)
        >>> cl.plot_find()
        >>> cl.scatter()

        """
        # If face detection, grayscale should be True.
        if (not self.grayscale): logger.warning('It is advisable to set "grayscale=True" when detecting faces.')

        # Read and pre-proces the input images
        logger.info("Read images>")
        # Create empty list
        faces = {'img':[], 'pathnames':[], 'filenames':[], 'facepath':[], 'coord_faces':[], 'coord_eyes':[]}
        # Extract faces and eyes from image
        for pathname in pathnames:
            # Extract faces
            facepath, imgfaces, coord_faces, coord_eyes, filename, path_to_image = self._extract_faces(pathname)
            # Store
            faces['pathnames'].append(path_to_image)
            faces['filenames'].append(filename)
            faces['facepath'].append(facepath)
            faces['img'].append(imgfaces)
            faces['coord_faces'].append(coord_faces)
            faces['coord_eyes'].append(coord_eyes)

        # Return
        self.results_faces = faces
        return faces

    def preprocessing(self, pathnames, grayscale, dim, flatten=True):
        """Pre-processing the input images and returning consistent output.

        Parameters
        ----------
        pathnames : list of str.
            Full path to images that are used in the model.
        grayscale : Bool, (default: False)
            Colorscaling the image to gray. This can be usefull when clustering e.g., faces.
        dim : tuple, (default: (128,128))
            Rescale images. This is required because the feature-space need to be the same across samples.
        flatten : Bool, (default: True)
            Flatten the processed NxMxC array to a 1D-vector

        Returns
        -------
        Xraw : dict containing keys:
            img : array-like.
            pathnames : list of str.
            filenames : list of str.

        """
        logger.info("Reading images..")
        img, filenames = None, None
        if isinstance(pathnames, str):
            pathnames=[pathnames]
        if isinstance(pathnames, list):
            filenames = list(map(basename, pathnames))
            img = list(map(lambda x: self.img_read_pipeline(x, grayscale=grayscale, dim=dim, flatten=flatten), tqdm(pathnames)))
            if flatten: img = np.vstack(img)

        out = {}
        out['img'] = img
        out['pathnames'] = pathnames
        out['filenames'] = filenames
        return out

    def extract_hog(self, X, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
        """Extract HOG features.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.

        Returns
        -------
        feat : array-like
            NxF array for which N are the samples and F the reduced feature space.

        """
        # If only a single image, make 2D-array
        ToSingle=False
        if len(X.shape)==1:
            X=np.c_[X, X].T
            ToSingle=True

        # Set dim correctly for reshaping image
        dim = _check_dim(X, self.dim, grayscale=self.grayscale)
        # Extract hog features per image
        feat = list(map(lambda x: hog(x.reshape(dim), orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)[1].flatten(), tqdm(X)))
        # Stack all hog features into one array and return
        feat = np.vstack(feat)
        # Back to single image if that was the input
        if ToSingle: feat=feat[0,:]
        # Return
        return feat

    def extract_pca(self, X):
        """Extract Principal Components.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.

        Returns
        -------
        feat : array-like
            NxF array for which N are the samples and F the reduced feature space.

        """
        # Check whether n_components is ok
        if self.params_pca['n_components']>X['img'].shape[0]:
            logger.warning('n_components should be smaller then the number of samples: %s<%s. Set as following during init: params_pca={"n_components":%s} ' %(X['img'].shape[0], self.params_pca['n_components'], X['img'].shape[0]))
            self.params_pca['n_components'] = X['img'].shape[0]

        # Fit model using PCA
        self.pca = pca(**self.params_pca)
        self.pca.fit_transform(X['img'], row_labels=X['filenames'])
        # Return
        return self.pca.results['PC'].values

    def _import_data(self, Xraw):
        """Import images and return in an consistent manner.

        Description
        -----------
        The input for the fit_transform() can have multiple forms; path to directory, list of strings and and array-like input.
        This requires that each of the input needs to be processed in its own manner but each should return the same structure to make it compatible across all functions.
        The following steps are used for the import:
            1. Images are imported with specific extention (['png','tiff','jpg']), 
            2. Each input image can then be grayscaled. Setting the grayscale parameter to True can be especially usefull when clustering faces.
            3. Final step in pre-processing is resizing all images in the same dimension such as (128,128). Note that if an array-like dataset [Samples x Features] is given as input, setting these dimensions are required to restore the image in case of plotting.
            4. Images are saved to disk in case a array-like input is given.
            5. Independent of the input, a dict is returned in a consistent manner.

        Processing the input depends on the input:

        Parameters
        ----------
        Xraw : str, list or array-like.
            The input can be:
                * "c://temp//" : Path to directory with images
                * ['c://temp//image1.png', 'c://image2.png', ...] : List of exact pathnames.
                * [[.., ..], [.., ..], ...] : Array-like matrix in the form of [sampels x features]

        Returns
        -------
        Object.
        model : dict
            dict containing keys with results.
            img : array-like.
                Pre-processed images
            pathnames : list of str.
                Full path to images that are used in the model.
            filenames : list of str.
                Filename of the input images.

        """
        # Check whether input is directory, list or array-like
        # 1. Collect images from directory
        if isinstance(Xraw, str) and os.path.isdir(Xraw):
            logger.info('Extracting images from: [%s]', Xraw)
            Xraw = self.get_images_from_path(Xraw, ext=self.ext)
            logger.info('Extracted images: [%s]', len(Xraw))

        # 2. Read images
        if isinstance(Xraw, list):
            # Make sure that list in lists are flattend
            Xraw = list(np.hstack(Xraw))
            # Read images and preprocessing
            X = self.preprocessing(Xraw, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=True)

        # 3. If input is array-like. Make sure X becomes compatible.
        if isinstance(Xraw, np.ndarray):
            # Make 2D
            if len(Xraw.shape)==1:
                Xraw = Xraw.reshape(-1,1).T
            # Check dimensions
            pathnames, filenames = None, None
            # Check dim
            self.dim = _check_dim(Xraw, self.dim)
            # Store to disk
            if self.store_to_disk:
                pathnames, filenames = store_to_disk(Xraw, self.dim, self.tempdir)

            # Make dict
            X = {'img': Xraw, 'pathnames':pathnames, 'filenames':filenames}
        return X

    def _clean(self):
        """Clean or removing previous results and models to ensure correct working."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if hasattr(self, 'results'): del self.results
            if hasattr(self, 'results_faces'): del self.results_faces
            if hasattr(self, 'distfit'): del self.distfit
            if hasattr(self, 'clusteval'): del self.clusteval
            if hasattr(self, 'pca'): del self.pca

    def _compute_embedding(self, X):
        """Compute the embedding for the extracted features.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.

        Returns
        -------
        xycoord : array-like.
            x,y coordinates after embedding or alternatively the first 2 features.
        """
        # Embedding using tSNE
        if self.embedding=='tsne':
            logger.info('Computing embedding using %s..', self.embedding)
            init='random'
            # if self.method is None: init='pca'
            xycoord = TSNE(n_components=2, learning_rate='auto', init=init).fit_transform(X)
        else:
            xycoord = X[:,0:2]
        # Return
        return xycoord

    def _extract_feat(self, Xraw):
        """Extract features based on the input data X.

        Parameters
        ----------
        Xinput : dict containing keys:
            img : array-like.
            pathnames : list of str.
            filenames : list of str.

        Returns
        -------
        Xraw : dict containing keys:
            img : array-like.
            pathnames : list of str.
            filenames : list of str.
        X : array-like
            Extracted features.

        """
        # If the input is a directory, first collect the images from path
        logger.info('Extracting features using method: [%s]', self.method)
        # Extract features
        if self.method=='pca':
            X = self.extract_pca(Xraw)
        elif self.method=='hog':
            X = self.extract_hog(Xraw['img'], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        else:
            # Read images and preprocessing and flattening of images
            X = Xraw['img'].copy()

        # Message
        logger.info("Extracted features using [%s]: %s" %(self.method, str(X.shape)))
        return Xraw, X

    def _compute_distances_pca(self, X, metric, alpha):
        """Compute distances and probabilities for new unseen samples.
        
        Description
        ------------
        In case of PCA, a transformation needs to take place first.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'cityblock', 'correlation', 'cosine', 'euclidean', 'hamming', 'jaccard', etc
        alpha : float, default: 0.05
            Significance alpha.

        """
        # Transform new unseen datapoint into feature space
        PCnew = self.pca.transform(X['img'], row_labels=X['filenames'])

        # Compute distance to all samples
        Y = distance.cdist(self.results['feat'], PCnew, metric=metric)
        Ytot = distance.cdist(self.results['feat'], self.results['feat'], metric=metric)

        # Fit distribution to emperical data and compute probability of the distances of interest
        if (alpha is not None) and (not hasattr(self, 'distfit')):
            # Take a subset of samples to prevent high computation times.
            x_max, y_max = np.minimum(500, Ytot.shape[0]), np.minimum(500, Ytot.shape[1])
            xrow, yrow = random.sample(range(1, x_max), x_max-1), random.sample(range(1, y_max), y_max-1)
            # Init distfit
            self.distfit = distfit(bound='down', multtest=None, distr=['norm', 'expon', 'uniform', 'gamma', 't'])
            Ytot = Ytot[xrow, :]
            Ytot = Ytot[:, yrow]
            _ = self.distfit.fit_transform(Ytot)
            # dist.plot()
        else:
            logger.info('Loading pre-fitted theoretical model..')

        # Sanity check
        if len(X['filenames'])!=Y.shape[1]: raise Exception(logger.error('Number of input files does not match number of computed distances.'))
        # Return
        return Y, PCnew

    def _extract_faces(self, pathname):
        """Extract the faces from the image.

        Parameters
        ----------
        pathname : str.
            Full path to a single image.

        Returns
        -------
        facepath : list of str.
            Filename of the extracted faces that are stored to disk.
        img : array-like.
            NxMxC for which N are the Samples, M the features and C the number of channels.
        pathnames : list of str.
            Full path to images that are used in the model.
        coord_faces : array-like.
            list of lists containing coordinates fo the faces in the original image.
        coord_eyes : array-like.
            list of lists containing coordinates fo the eyes in the extracted (img and facepath) image.
        filenames : list of str.
            Filename of the input images.
        pathnames : list of str.
            Pathnames of the input images.

        """
        # Set defaults
        coord_eyes, facepath, imgstore = [], [], []
        # Get image
        X = self.preprocessing(pathname, grayscale=self.cv2_imread_colorscale, dim=None, flatten=False)
        # Get the image and Convert into grayscale if required
        img = X['img'][0]
        # img = to_gray(X['img'][0])
        # Detect faces using the face_cascade
        coord_faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

        # Collect the faces from the image
        for (x,y,w,h) in coord_faces:
            # Create filename for face
            filename = os.path.join(self.dirpath, str(uuid.uuid4()))+'.png'
            facepath.append(filename)
            # Store faces seperately
            imgface = img_resize(img[y:y+h, x:x+w], dim=self.dim_face)
            # Write to disk
            cv2.imwrite(filename, imgface)
            # Store face image
            # imgstore.append(imgface.flatten())
            imgstore.append(img_flatten(imgface))
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(imgface)
            if eyes==(): eyes=None
            coord_eyes.append(eyes)
        # Return
        return facepath, np.array(imgstore), coord_faces, coord_eyes, X['filenames'][0], X['pathnames'][0]

    def _collect_pca(self, X, Y, k, alpha, feat, todf=True):
        """Collect the samples that are closest in according the metric."""

        filenames = X['filenames']
        out = {}
        out['feat'] = feat

        # Collect nearest neighor and sample with highes probability per input sample
        for i, filename in enumerate(filenames):
            logger.info('Find: %s', filename)
            store_key = {}
            idx_dist, idx_k = None, None
            # Collect bes samples based on k-nearest neighbor
            if k is not None:
                idx_k = np.argsort(Y[:,i])[0:k]
            # Collect samples based on probability
            if alpha is not None:
                dist_results = self.distfit.predict(Y[:,i], verbose=0)
                idx_dist = np.where(dist_results['y_proba']<=alpha)[0]
            else:
                # If alpha is not used, set all to nan
                dist_results={}
                dist_results['y_proba'] = np.array([np.nan]*Y.shape[0])

            # Combine the unique k-nearest samples and probabilities.
            idx = unique_no_sort(np.append(idx_dist, idx_k))

            # Store in dict
            store_key = {**store_key, 'x_path': X['pathnames'][i], 'y_idx': idx, 'distance': Y[idx, i], 'y_proba': dist_results['y_proba'][idx], 'y_label': np.array(self.results['filenames'])[idx].tolist(), 'y_path': np.array(self.results['pathnames'])[idx].tolist()}
            if todf: store_key = pd.DataFrame(store_key)
            out[filename] = store_key

        # Return
        return out

    def img_read_pipeline(self, filepath, grayscale=1, dim=(128, 128), flatten=True):
        """Read and pre-processing of images.
    
        Parameters
        ----------
        filepath : str
            Full path to the image that needs to be imported.
        grayscale : int, default: 1 (gray)
            colour-scaling from opencv.
            * cv2.COLOR_GRAY2RGB
        dim : tuple, (default: (128,128))
            Rescale images. This is required because the feature-space need to be the same across samples.
        flatten : Bool, (default: True)
            Flatten the processed NxMxC array to a 1D-vector
    
        Returns
        -------
        img : array-like
            Imported and processed image.
    
        """
        # Read the image
        img = img_read(filepath, grayscale=grayscale)
        # Scale the image
        img = img_scale(img)
        # Resize the image
        img = img_resize(img, dim=dim)
        # Flatten the image
        if flatten: img = img_flatten(img)
        return img

    def save(self, filepath='clustimage.pkl', overwrite=False, verbose=3):
        """Save model in pickle file.
    
        Parameters
        ----------
        filepath : str, (default: 'clustimage.pkl')
            Pathname to store pickle files.
        overwrite : bool, (default=False)
            Overwite file if exists.
        verbose : int, optional
            Show message. A higher number gives more informatie. The default is 3.
    
        Returns
        -------
        bool : [True, False]
            Status whether the file is saved.
    
        """
        if (filepath is None) or (filepath==''):
            filepath = 'clustimage.pkl'
        if filepath[-4:] != '.pkl':
            filepath = filepath + '.pkl'
        # Store data
        storedata = {}
        storedata['results'] = self.results
        storedata['results_faces'] = self.results_faces
        storedata['distfit'] = self.distfit
        storedata['clusteval'] = self.clusteval
        storedata['pca'] = self.pca
        storedata['params'] = self.params
        storedata['params_pca'] = self.params_pca
        storedata['params_hog'] = self.params_hog
        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=verbose)
        logger.info('Saving..')
        # return
        return status

    def load(self, filepath='clustimage.pkl', verbose=3):
        """Restore previous results.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        Object.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'clustimage.pkl'
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'

        # Load
        storedata = pypickle.load(filepath, verbose=verbose)

        # Store in self
        if storedata is not None:
            self.results = storedata['results']
            self.results_faces = storedata['results_faces']
            self.distfit = storedata['distfit']
            self.clusteval = storedata['clusteval']
            self.pca = storedata['pca']
            self.params = storedata['params']
            self.params_pca = storedata['params_pca']
            self.params_hog = storedata['params_hog']

            logger.info('Load succesful!')
            # Return results
            return self.results
        else:
            logger.warning('Could not load previous results!')


    def plot_faces(self, faces=True, eyes=True, cmap=None):
        """Plot detected faces.

        Description
        -----------
        Plot the detected faces in images after using the fit_transform() function.
        * For each input image, rectangles are drawn over the detected faces.
        * Each face is plotted seperately for which rectlangles are drawn over the detected eyes.

        Parameters
        ----------
        faces : Bool, (default: True)
            Plot the seperate faces.
        eyes : Bool, (default: True)
            Plot rectangles over the detected eyes.
        cmap : str, (default: None)
            Colorscheme for the images.
                * 'gray'
                * 'binary'
                * None : uses rgb colorscheme

        """
        cmap = _set_cmap(cmap, self.grayscale)

        # Walk over all detected faces
        if hasattr(self, 'results_faces'):
            for i, pathname in enumerate(self.results_faces['pathnames']):
                # Import image
                img = self.preprocessing(pathname, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()

                # Plot the faces
                if faces:
                    coord_faces = self.results_faces['coord_faces'][i]
                    plt.figure()
                    for (x,y,w,h) in coord_faces:
                        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    if len(img.shape)==3:
                        plt.imshow(img[:,:,::-1], cmap=cmap) # RGB-> BGR
                    else:
                        plt.imshow(img, cmap=cmap)

                # Plot the eyes
                if eyes:
                    coord_eyes = self.results_faces['coord_eyes'][i]
                    for k in np.arange(0, len(self.results_faces['facepath'][i])):
                        # face = self.results_faces['img'][i][k].copy()
                        facepath = self.results_faces['facepath'][i][k]
                        if os.path.isfile(facepath):
                            face = self.preprocessing(facepath, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()
                            if coord_eyes[k] is not None:
                                plt.figure()
                                for (ex,ey,ew,eh) in coord_eyes[k]:
                                    cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                                if len(face.shape)==3:
                                    plt.imshow(face[:,:,::-1]) # RGB-> BGR
                                else:
                                    plt.imshow(face)
                        else:
                            logger.warning('File is removed: %s', facepath)
        else:
            logger.warning('Nothing to plot. First detect faces with ".detect_faces(pathnames)"')

    def dendrogram(self, max_d=None, figsize=(15,10)):
        """Plot Dendrogram.

        Parameters
        ----------
        max_d : Float, (default: None)
            Height of the dendrogram to make a horizontal cut-off line.
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).

        Returns
        -------
        results : list
            Cluster labels.

        Returns
        -------
        None.

        """
        if hasattr(self, 'clusteval'):
            self.clusteval.plot()
            results = self.clusteval.dendrogram(max_d=max_d, figsize=figsize)
            # if feat is None: feat=self.results['feat']
            # results = self.clusteval.dendrogram(X=feat, max_d=max_d, figsize=figsize)
        return results

    def scatter(self, dotsize=15, legend=False, figsize=(15,10)):
        """Plot the samples using a scatterplot.

        Parameters
        ----------
        dotsize : int, (default: 15)
            Dot size of the scatterpoints.
        legend : bool, (default: False)
            Plot the legend.
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        # Set default settings
        labx = self.results.get('labx', None)
        if labx is None: labx=np.zeros_like(self.results['xycoord'][:,0]).astype(int)

        # Scatter cluster evaluation
        if hasattr(self, 'clusteval'):
            if self.cluster_space=='low':
                xycoord = self.results['xycoord']
            else:
                xycoord = self.results['feat']
            # Scatter
            self.clusteval.scatter(xycoord)

        if self.embedding=='tsne':
            colours=np.vstack(colourmap.fromlist(labx)[0])
            title = ('tSNE plot for which the samples are coloured on the cluster-labels of the the [%s] feature space.' %(self.cluster_space))
            fig, ax = scatterd(self.results['xycoord'][:,0], self.results['xycoord'][:,1], s=dotsize, c=colours, label=labx, figsize=figsize, title=title, fontsize=18, fontcolor=[0,0,0])

        # Scatter all points
        if self.method=='pca':
            _, ax = self.pca.plot(figsize=figsize)
            _, ax = self.pca.scatter(y=labx, legend=legend, label=False, figsize=figsize)

        # Scatter the predicted cases
        if self.results.get('predict', None) is not None:
            fig, ax = self.pca.scatter(y=labx, legend=legend, label=False, figsize=figsize)
            # Create unique colors
            colours = colourmap.fromlist(self.results['predict']['feat'].index)[1]
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    x,y = self.results['predict']['feat'].iloc[:,0:2].loc[key]
                    idx = self.results['predict'][key]['y_idx']
                    # Scatter
                    ax.scatter(x, y, color=colours[key], edgecolors=[0,0,0])
                    ax.text(x,y, key, color=colours[key])
                    ax.scatter(self.results['feat'][idx][:,0], self.results['feat'][idx][:,1], edgecolors=[0,0,0])

    def plot_find(self, cmap=None, figsize=(15,10)):
        """Plot the input image together with the predicted images.

        Parameters
        ----------
        cmap : str, (default: None)
            Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        cmap = _set_cmap(cmap, self.grayscale)
        # Plot the images that are similar to each other.
        if self.results.get('predict', None) is not None:
            for key in self.results['predict'].keys():

                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    logger.info('Plotting results for [%s]', key)
                    # Collect images
                    input_img = self.results['predict'][key]['x_path']
                    find_img = self.results['predict'][key]['y_path']
                    # Input label
                    if isinstance(input_img, str): input_img=[input_img]
                    # Input images
                    I_input = list(map(lambda x: self.img_read_pipeline(x, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False), input_img))
                    # Predicted label
                    I_find = list(map(lambda x: self.img_read_pipeline(x, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False), find_img))
                    # Make the real plot
                    title='Top or top-left image is input. The others are predicted.'
                    # Show images into subplots
                    imgs=I_input+I_find
                    self._make_subplots(imgs, None, cmap, figsize, title)

    def plot(self, labx=None, ncols=10, cmap=None, show_hog=False, figsize=(15,10)):
        """Plot the results.

        Parameters
        ----------
        labx : int, (default: None)
            Cluster label to plot. In case of None, all cluster labels are plotted.
        ncols : int, (default: 10)
            Number of columns to use in the subplot. The number of rows are estimated based on the columns.
        Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        show_hog : bool, (default: False)
            Plot the hog features next to the input image.
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        # Set cmap
        cmap = _set_cmap(cmap, self.grayscale)
        # Plot the clustered images
        if (self.results.get('labx', None) is not None) and (self.results.get('pathnames', None) is not None):
            if labx is None:
                uilabx = np.unique(self.results['labx'])
            else:
                uilabx = [labx]
            
            # Run over all labels.
            for labx in tqdm(uilabx):
                idx = np.where(self.results['labx']==labx)[0]
                # Collect the images
                getfiles = np.array(self.results['pathnames'])[idx]
                # Get the images that cluster together
                imgs = list(map(lambda x: self.img_read_pipeline(x, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False), getfiles))
                # Make subplots
                self._make_subplots(imgs, ncols, cmap, figsize, ("Images in cluster %s" %(str(labx))))

                # Make hog plots
                if show_hog and (self.method=='hog'):
                    hog_images = self.results['feat'][idx,:]
                    fig, axs = plt.subplots(len(imgs), 2, figsize=(15,10), sharex=True, sharey=True)
                    for i, ax in enumerate(axs):
                        hog_image_rescaled = exposure.rescale_intensity(hog_images[i,:].reshape(self.dim), in_range=(0,10))
                        ax[0].imshow(imgs[i], cmap=plt.cm.gray)
                        ax[0].axis('off')
                        ax[1].imshow(hog_image_rescaled, cmap=plt.cm.gray)
                        ax[1].axis('off')

                    _ = fig.suptitle('Histogram of Oriented Gradients', fontsize=16)
                    plt.tight_layout()
                    plt.show()
        else:
            logger.warning('Plotting is not possible if path locations are unknown. Your input may have been a data-array. Try to set "store_to_disk=True" during initialization.')

    def _make_subplots(self, imgs, ncols, cmap, figsize, title=''):
        """Make subplots."""

        dim = self.dim if self.grayscale else np.append(self.dim, 3)

        if ncols is None:
            ncols = 5
            if len(imgs)>25: ncols=10
            if len(imgs)>=100: ncols=15
            if len(imgs)>=150: ncols=20
        
        # Setup rows and columns
        nrows = int(np.ceil(len(imgs)/ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, ax in enumerate(axs.ravel()):
            if i<len(imgs):
                ax.imshow(imgs[i].reshape(dim), cmap=cmap)
            ax.axis("off")
        _ = fig.suptitle(title, fontsize=16)
        plt.pause(0.1)

    def get_images_from_path(self, dirpath, ext=['png','tiff','jpg']):
        return _get_images_from_path(dirpath, ext=ext)
    
    def clean_files(self):
        # Cleaning
        from pathlib import Path
        if hasattr(self, 'results_faces'):
            out = []
            for sublist in self.results_faces['facepath']:
                out.extend(sublist)

            p = Path(out[0])
            dirpath = str(p.parent)

            if os.path.isdir(dirpath):
                logger.info('Removing directory with all content: %s', dirpath)
                shutil.rmtree(dirpath)
        else:
            logger.info('Nothing to clean.')

    def import_example(self, data='flowers', url=None):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            'flowers', 'faces', 'scenes'

        Returns
        -------
        list of str
            list of str containing filepath to images.

        """
        return import_example(data=data, url=url)


# %% Store images to disk
def _check_dim(Xraw, dim, grayscale=None):
    dimOK=False
    # Determine the dimension based on the length of the 1D-vector.
    # if len(Xraw.shape)==1:
        # Xraw=np.c_[Xraw, Xraw].T
    if len(Xraw.shape)==1:
        Xraw = Xraw.reshape(-1,1).T

    # Compute dim based on vector length
    dimX = int(np.sqrt(Xraw.shape[1]))

    if (dimX!=dim[0]) or (dimX!=dim[1]):
        logger.warning('The default dim=%s of the image does not match with the input: %s. Set dim=%s during initialization!' %(str(dim), str([int(dimX)]*2), str([int(dimX)]*2) ))

    if not dimOK:
        try:
            Xraw[0,:].reshape(dim)
            dimOK=True
        except:
            pass

    if not dimOK:
        try:
            Xraw[0,:].reshape(np.append(dim, 3))
            dim=np.append(dim, 3)
            dimOK=True
        except:
            pass

    if not dimOK:
        try:
            Xraw[0,:].reshape([dimX, dimX])
            dim = [dimX, dimX]
            dimOK=True
        except:
            pass

    if not dimOK:
        try:
            Xraw[0,:].reshape([dimX, dimX, 3])
            dim = [dimX, dimX, 3]
        except:
            pass

    if not dimOK:
        raise Exception(logger.error('The default dim=%s of the image does not match with the input: %s. Set dim=%s during initialization!' %(str(dim), str([int(dimX)]*2), str([int(dimX)]*2) )))
    else:
        logger.info('The dim is changed into: %s', str(dim))

    return dim

# %% Store images to disk
def store_to_disk(Xraw, dim, tempdir):
    """Store to disk."""
    # Determine the dimension based on the length of the vector.
    dim = _check_dim(Xraw, dim)
    # Store images to disk
    pathnames, filenames = [], []
    logger.info('Writing images to tempdir [%s]', tempdir)
    for i in tqdm(np.arange(0, Xraw.shape[0])):
        filename = str(uuid.uuid4())+'.png'
        pathname = os.path.join(tempdir, filename)
        # Write to disk
        cv2.imwrite(pathname, Xraw[i,:].reshape(dim))
        filenames.append(filename)
        pathnames.append(pathname)
    return pathnames, filenames

# %% Unique without sort
def unique_no_sort(x):
    """Unique without sort."""
    x = x[x!=None]
    indexes = np.unique(x, return_index=True)[1]
    return [x[index] for index in sorted(indexes)]

# %% Resize image
def basename(label):
    """Extract basename from path."""
    return os.path.basename(label)

# %% Resize image
def img_flatten(img):
    """Flatten image."""
    return img.flatten()

# %% Resize image
def img_resize(img, dim=(128, 128)):
    """Resize image."""
    if dim is not None:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

# %% Set cmap
def _set_cmap(cmap, grayscale):
    """Set the colourmap."""
    if cmap is None:
        cmap = 'gray' if grayscale else None
    return cmap

# %% Scaling
def img_scale(img):
    """Normalize image by scaling.

    Description
    -----------
    Scaling in range [0-255] by img*(255/max(img))

    Parameters
    ----------
    img : array-like
        Input image data.

    Returns
    -------
    img : array-like
        Scaled image.
    """
    try:
        # Normalizing between 0-255
        img = img - img.min()
        img = img / img.max()
        img = img * 255
        # Downscale typing
        img = np.uint8(img)
    except:
        logger.warning('Scaling not possible.')
    return img


# %% Read image
def img_read(filepath, grayscale=1):
    """Read image from filepath using colour-scaling.

    Parameters
    ----------
    filepath : str
        path to file.
    grayscale : int, default: 1 (gray)
        colour-scaling from opencv.
        * cv2.COLOR_GRAY2RGB

    Returns
    -------
    img : numpy array
        raw rgb or gray image.

    """
    img=None
    if os.path.isfile(filepath):
        # Read the image
        img = cv2.imread(filepath, grayscale)
    else:
        logger.warning('File does not exists: %s', filepath)
    
    # In case of rgb images: make gray images compatible with RGB
    if ((grayscale!=0) and (grayscale!=6)) and (len(img.shape)<3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

# %%
def set_logger(verbose=20):
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %% Import example dataset from github.
def import_example(data='flowers', url=None):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'flowers', 'faces', 'digits'
    url : str
        url link to to dataset.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='flowers':
            url='https://erdogant.github.io/datasets/flower_images.zip'
        elif data=='faces':
            url='https://erdogant.github.io/datasets/faces_images.zip'
        elif data=='scenes':
            url='https://erdogant.github.io/datasets/scenes.zip'
        elif data=='digits':
            from sklearn.datasets import load_digits
            digits = load_digits(n_class=10)
            # y = digits.target
            return digits.data  
    else:
        logger.warning('Lets try your dataset from url: %s.', url)

    if url is None:
        logger.warning('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = basename(urlparse(url).path)
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
    image_files = _get_images_from_path(dirpath)
    # Return
    return image_files


# %% Recursively list files from directory
def _get_images_from_path(dirpath, ext=['png','tiff','jpg']):
    """ Recursively collect images from path.

    Parameters
    ----------
    dirpath : str
        Path to directory; "/tmp" or "c://temp/" 
    ext : list, default: ['png','tiff','jpg']
        extentions to collect form directories.

    Returns
    -------
    getfiles : list of str.
        Full pathnames to images.

    """
    if not isinstance('dirpath', str): raise Exception(print('Error: "dirpath" should be of type string.'))
    if not os.path.isdir(dirpath): raise Exception(print('Error: The directory can not be found: %s.' %dirpath))
    
    getfiles = []
    for iext in ext:
        for root, _, filenames in os.walk(dirpath):
            for filename in fnmatch.filter(filenames, '*.'+iext):
                getfiles.append(os.path.join(root, filename))
    logger.info('[%s] files are collected recursively from path: [%s]', len(getfiles), dirpath)
    return getfiles


# %% unzip
def _unzip(path_to_zip):
    """Unzip files.

    Parameters
    ----------
    path_to_zip : str
        Path of the zip file.

    Returns
    -------
    getpath : str
        Path containing the unzipped files.

    """
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
    """ Retrieve file from url.

    Parameters
    ----------
    url : str.
        Internet source.
    writepath : str.
        Directory to write the file.

    Returns
    -------
    None.

    """
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
