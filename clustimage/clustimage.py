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
from ismember import ismember
import colourmap
import datazets as dz
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
# from umap.umap_ import UMAP
import os
import logging
import fnmatch
import cv2
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.spatial import distance
from skimage.feature import hog
from skimage import exposure
import tempfile
import uuid
import shutil
import random
import imagehash
from PIL import Image

# Configure the logger
logger = logging.getLogger('')
[logger.removeHandler(handler) for handler in logger.handlers[:]]
console = logging.StreamHandler()
formatter = logging.Formatter('[clustimage] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Clustimage():
    """Clustering of images.

    Clustering input images after following steps of pre-processing, feature-extracting, feature-embedding and cluster-evaluation.
    Taking all these steps requires setting various input parameters. Not all input parameters can be changed across the different steps in clustimage.
    Some parameters are choosen based on best practice, some parameters are optimized, while others are set as a constant.

    The following 4 steps are taken:

    * Step 1. Pre-processing.
        Images are imported with specific extention (['png','tiff','jpg']),
        Each input image can then be grayscaled. Setting the grayscale parameter to True can be especially usefull when clustering faces.
        Final step in pre-processing is resizing all images in the same dimension such as (128,128). Note that if an array-like dataset [Samples x Features] is given as input, setting these dimensions are required to restore the image in case of plotting.
    * Step 2. Feature-extraction.
        Features are extracted from the images using Principal component analysis (PCA), Histogram of Oriented Gradients (HOG) or the raw values are used.
    * Step 3. Embedding:
        The feature-space non-lineair transformed using t-SNE and the coordinates are stored. The embedding is only used for visualization purposes.
    * Step 4. Cluster evaluation.
        The feature-space is used as an input in the cluster-evaluation method. The cluster evaluation method determines the optimal number of clusters and return the cluster labels.
    * Step 5: Save.
        The results are stored in the object and returned by the model. Various different (scatter) plots can be made to evaluate the results.

    Parameters
    ----------
    method : str, (default: 'pca')
        Method to be usd to extract features from images.
            * None : No feature extraction
            * 'pca' : PCA feature extraction
            * 'hog' : hog features extraced
            * 'pca-hog' : PCA extracted features from the HOG desriptor
            hashmethod : str (default: 'ahash')
            * 'ahash': Average hash
            * 'phash': Perceptual hash
            * 'dhash': Difference hash
            * 'whash-haar': Haar wavelet hash
            * 'whash-db4': Daubechies wavelet hash
            * 'colorhash': HSV color hash
            * 'crop-resistant-hash': Crop-resistant hash
    embedding : str, (default: 'tsne')
        Perform embedding on the extracted features. The xycoordinates are used for plotting purposes. For UMAP; all default settings are used, and with densmap=True.
            * 'tsne'
            * 'umap'
            * None
    grayscale : Bool, (default: False)
        Colorscaling the image to gray. This can be usefull when clustering e.g., faces.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    dirpath : str, (default: 'clustimage')
        Directory to write images. The default is the system tempdirectory.
    ext : list, (default: ['png','tiff','jpg'])
        Images with the file extentions are used.
    params_pca : dict, default: {'n_components':50, 'detect_outliers':None}
        Parameters to initialize the pca model.
    params_hog : dict, default: {'orientations':9, 'pixels_per_cell':(16,16), 'cells_per_block':(1,1)}
        Parameters to extract hog features.
    verbose : int, (default: 'info')
        Print progress to screen. The default is 20.
        60: None, 40: error, 30: warning, 20: info, 10: debug

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
        labels : list.
            Cluster labels

    Example
    -------
    >>> from clustimage import Clustimage
    >>>
    >>> # Init with default settings
    >>> cl = Clustimage(method='pca')
    >>>
    >>> # load example with faces
    >>> X, y = cl.import_example(data='mnist')
    >>>
    >>> # Cluster digits
    >>> results = cl.fit_transform(X)
    >>>
    >>> # Cluster evaluation
    >>> cl.clusteval.plot()
    >>> cl.clusteval.scatter(cl.results['xycoord'])
    >>> cl.clusteval.plot_silhouette(cl.results['xycoord'])
    >>> cl.pca.plot()
    >>>
    >>> # Unique
    >>> cl.plot_unique(img_mean=False)
    >>> cl.results_unique.keys()
    >>>
    >>> # Scatter
    >>> cl.scatter(img_mean=False, zoom=3)
    >>> cl.scatter(zoom=8, plt_all=True, figsize=(150,100))
    >>>
    >>> # Plot clustered images
    >>> cl.plot(labels=8)
    >>>
    >>> # Plot dendrogram
    >>> cl.dendrogram()
    >>>
    >>> # Find images
    >>> results_find = cl.find(X[0,:], k=None, alpha=0.05)
    >>> cl.plot_find()
    >>> cl.scatter()
    >>>

    """

    def __init__(self,
                 method='pca',
                 embedding='tsne',
                 grayscale=False,
                 dim=(128, 128),
                 dim_face=(64, 64),
                 dirpath=None,
                 store_to_disk=True,
                 ext=['png', 'tiff', 'jpg'],
                 params_pca={'n_components': 0.95},
                 params_hog={'orientations': 8, 'pixels_per_cell': (8, 8), 'cells_per_block': (1, 1)},
                 params_hash={'threshold': 0, 'hash_size': 8},
                 verbose='info'):
        """Initialize clustimage with user-defined parameters."""
        # Clean readily fitted models to ensure correct results
        self.clean_init()

        if not (np.any(np.isin(method, [None, 'pca', 'hog', 'pca-hog'])) or ('hash' in method)): raise Exception(logger.error('method: "%s" is unknown', method))
        # Check method types
        if (np.any(np.isin(method, ['hog', 'pca-hog']))) and ~grayscale:
            logger.warning('Parameter grayscale is set to True because you are using method="%s"' %(method))
            grayscale=True
        if (dim is None) or ((dim[0]>1024) or (dim[1]>1024)):
            logger.warning('Setting dim > (1024, 1024) is most often not needed and can cause memory and other issues.')
        if method=='crop-resistant-hash':
            logger.info('Hash size is set to 8 for crop-resistant and can not be changed.')
            params_hash['hash_size'] = 8
        if method=='whash-haar':
            if (np.ceil(np.log2(params_hash['hash_size'])) != np.floor(np.log2(params_hash['hash_size']))):
                logger.error('hash_size should be power of 2 (8, 16, 32, 64, ..., etc)')
                return None

        # Find path of xml file containing haarcascade file and load in the cascade classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.params = {}
        self.params['cv2_imread_colorscale'] = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        self.params['face_cascade'] = 'cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")'
        self.params['eye_cascade'] = 'cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")'

        self.params['method'] = method
        self.params['embedding'] = embedding
        self.params['grayscale'] = grayscale
        self.params['dim'] = dim
        self.params['dim_face'] = dim_face

        self.params['dirpath'] = _set_tempdir(dirpath)
        self.params['tempdir'] = _set_tempdir(None)
        self.params['filepath'] = os.path.join(_set_tempdir(None), 'clustimage.pkl')
        self.params['ext'] = ext
        self.params['store_to_disk'] = store_to_disk

        # Hash parameters
        self.params_hash = get_params_hash(method, params_hash)
        # PCA parameters
        pca_defaults = {'n_components': 0.95, 'detect_outliers': None, 'random_state': None}
        params_pca = {**pca_defaults, **params_pca}
        self.params_pca = params_pca
        # HOG parameters
        hog_defaults = {'orientations': 8, 'pixels_per_cell': (8, 8), 'cells_per_block': (1, 1)}
        params_hog = {**hog_defaults, **params_hog}
        self.params_hog = params_hog
        # Set the logger
        set_logger(verbose=verbose)
        # This value is set to True when the find functionality (cl.find) is used to make sure specified subroutines are used.
        self.find_func = False

    def fit_transform(self, X, cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=3, max_clust=25, cluster_space='high', black_list=None):
        """Group samples into clusters that are similar in their feature space.

        The fit_transform function allows to detect natural groups or clusters of images. It works using a multi-step proces of pre-processing, extracting the features, and evaluating the optimal number of clusters across the feature space.
        The optimal number of clusters are determined using well known methods suchs as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
        Based on the clustering results, the unique images are also gathered.

        Parameters
        ----------
        X : [str of list] or [np.array].
            The input can be:
                * "c://temp//" : Path to directory with images
                * ['c://temp//image1.png', 'c://image2.png', ...] : List of exact pathnames.
                * [[.., ..], [.., ..], ...] : np.array matrix in the form of [sampels x features]
        cluster : str, (default: 'agglomerative')
            Type of clustering.
                * 'agglomerative'
                * 'kmeans'
                * 'dbscan'
                * 'hdbscan'
        evaluate : str, (default: 'silhouette')
            Cluster evaluation method.
                * 'silhouette'
                * 'dbindex'
                * 'derivative'
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'cityblock'
                * 'correlation'
                * 'cosine'
                * 'jaccard'
                * 'mahalanobis'
                * 'seuclidean'
                * 'sqeuclidean'
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
                * 'ward'
                * 'single'
                * 'complete'
                * 'average'
                * 'weighted'
                * 'centroid'
                * 'median'
        min_clust : int, (default: 3)
            Number of clusters that is evaluated greater or equals to min_clust.
        max_clust : int, (default: 25)
            Number of clusters that is evaluated smaller or equals to max_clust.
        cluster_space: str, (default: 'high')
            Selection of the features that are used for clustering. This can either be on high or low feature space.
                * 'high' : Original feature space.
                * 'low' : Input are the xycoordinates that are determined by "embedding". Thus either tSNE coordinates or the first two PCs or HOGH features.
        black_list : list, (default: None)
            Exclude directory with all subdirectories from processing.
            * example: ['undouble']

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
            labels : list.
                Cluster labels

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage(method='pca', grayscale=True)
        >>>
        >>> # load example with faces
        >>> pathnames, y = cl.import_example(data='faces')
        >>> # Detect faces
        >>> face_results = cl.extract_faces(pathnames)
        >>>
        >>> # Cluster extracted faces
        >>> results = cl.fit_transform(face_results['pathnames_face'])
        >>>
        >>> # Cluster evaluation
        >>> cl.clusteval.plot()
        >>> cl.clusteval.scatter(cl.results['xycoord'])
        >>>
        >>> # Unique
        >>> cl.plot_unique(img_mean=False)
        >>> cl.results_unique.keys()
        >>>
        >>> # Scatter
        >>> cl.scatter(dotsize=50, img_mean=False)
        >>>
        >>> # Plot clustered images
        >>> cl.plot(labels=8)
        >>> # Plot facces
        >>> cl.plot_faces()
        >>>
        >>> # Plot dendrogram
        >>> cl.dendrogram()
        >>>
        >>> # Find images
        >>> results_find = cl.find(face_results['pathnames_face'][2], k=None, alpha=0.05)
        >>> cl.plot_find()
        >>> cl.scatter()
        >>> cl.pca.plot()
        >>>

        """
        # Cleaning
        self.clean_init()
        # Check whether in is dir, list of files or array-like
        _ = self.import_data(X, black_list=black_list)
        # Extract features using method
        _ = self.extract_feat(self.results)
        # Embedding
        _ = self.embedding(self.results['feat'], metric=metric)
        # Cluster
        self.cluster(cluster=cluster, evaluate=evaluate, cluster_space=cluster_space, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust)
        # Return
        return self.results

    def cluster(self, cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=3, max_clust=25, cluster_space='high'):
        """Detect the optimal number of clusters given the input set of features.

        This function is build on clusteval, which is a python package that provides various evalution methods for unsupervised cluster validation.

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
        evaluate : str, (default: 'silhouette')
            Cluster evaluation method.
                * 'silhouette'
                * 'dbindex'
                * 'derivative'
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'cityblock'
                * 'correlation'
                * 'cosine'
                * 'jaccard'
                * 'mahalanobis'
                * 'seuclidean'
                * 'sqeuclidean'
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
                * 'ward'
                * 'single'
                * 'complete'
                * 'average'
                * 'weighted'
                * 'centroid'
                * 'median'
        min_clust : int, (default: 3)
            Number of clusters that is evaluated greater or equals to min_clust.
        max_clust : int, (default: 25)
            Number of clusters that is evaluated smaller or equals to max_clust.

        Returns
        -------
        array-like
            .results['labels'] : Cluster labels.
            .clusteval : model parameters for cluster-evaluation and plotting.

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init
        >>> cl = Clustimage(method='hog')
        >>>
        >>> # load example with digits (mnist dataset)
        >>> pathnames = cl.import_example(data='flowers')
        >>>
        >>> # Find clusters
        >>> results = cl.fit_transform(pathnames)
        >>>
        >>> # Evaluate plot
        >>> cl.clusteval.plot()
        >>> cl.scatter(dotsize=50, img_mean=False)
        >>>
        >>> # Change the clustering evaluation approach, metric, minimum expected nr. of clusters etc.
        >>> labels = cl.cluster(min_clust=5, max_clust=25)
        >>>
        >>> # Evaluate plot
        >>> cl.clusteval.plot()
        >>> cl.scatter(dotsize=50, img_mean=False)
        >>>
        >>> # If you want to cluster on the low-dimensional space.
        >>> labels = cl.cluster(min_clust=5, max_clust=25, cluster_space='low', cluster='dbscan')
        >>> cl.scatter(dotsize=50, img_mean=False)
        >>>

        """
        if self.results.get('feat', None) is None: raise Exception(logger.error('First run the "fit_transform(pathnames)" function.'))
        self.params['cluster_space'] = cluster_space
        if min_clust==max_clust: max_clust=min_clust + 1
        ce = None

        if len(self.results['feat'])==0:
            return None

        ce = clusteval(cluster=cluster, evaluate=evaluate, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, verbose=get_logger())
        # Fit
        if cluster_space=='low':
            feat = self.results['xycoord']
            logger.info('Cluster evaluation using the [%s] feature space of the [%s] coordinates.', cluster_space, self.params['embedding'])
        else:
            feat = self.results['feat']
            logger.info('Cluster evaluation using the [%s] feature space of the [%s] features.', cluster_space, self.params['method'])
        # Fit model
        _ = ce.fit(feat)
        labels = ce.results['labx']
        logger.info('Updating cluster-labels and cluster-model based on the %s feature-space.', str(feat.shape))

        # Store results and params
        self.results['labels'] = labels
        self.params['cluster_space'] = cluster_space
        self.params['metric_find'] = metric
        self.clusteval = ce
        self.params_clusteval = {}
        self.params_clusteval['cluster'] = cluster
        self.params_clusteval['evaluate'] = evaluate
        self.params_clusteval['metric'] = metric
        self.params_clusteval['linkage'] = linkage
        self.params_clusteval['min_clust'] = min_clust
        self.params_clusteval['max_clust'] = max_clust

        # Find unique
        self.unique(metric=metric)

        # Return
        return self.results['labels']

    def _check_status(self):
        if not hasattr(self, 'results'):
            raise Exception(logger.error('Results in missing. Hint: fit_transform(X)'))

    def unique(self, metric=None):
        """Compute the unique images.

        The unique images are detected by first computing the center of the cluster, and then taking the image closest to the center.

        Parameters
        ----------
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'cityblock'
                * 'correlation'
                * 'cosine'
                * 'jaccard'
                * 'mahalanobis'
                * 'seuclidean'
                * 'sqeuclidean'
                * etc

        Returns
        -------
        dict containing keys with results.
            labels : list.
                Cluster label of the detected image.
            idx : list.
                Index of the original image.
            xycoord_center : array-like
                Coordinates of the sample that is most centered.
            pathnames : list.
                Path location to the file.
            img_mean : array-like.
                Averaged image in the cluster.

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage()
        >>>
        >>> # load example with faces
        >>> X, y = cl.import_example(data='mnist')
        >>>
        >>> # Cluster digits
        >>> _ = cl.fit_transform(X)
        >>>
        >>> # Unique
        >>> cl.plot_unique(img_mean=False)
        >>> cl.results_unique.keys()
        >>>

        """
        # Check status
        self._check_status()
        # Check whether embedding is performed
        if self.results['xycoord'] is None:
            logger.warning('Missing x,y coordinates in results dict. Hint: try to first run: cl.embedding(Xfeat)')
            return None

        if metric is None: metric=self.params_clusteval['metric']
        eigen_img, pathnames, center_idx, center_coord = [], [], [], []
        # Unique labels
        uilabels = np.unique(self.results['labels'])

        # Run over all cluster labels
        for label in uilabels:
            # Get cluster label
            idx = np.where(self.results['labels']==label)[0]
            # Compute center of cluster
            xycoord_center = np.mean(self.results['xycoord'][idx, :], axis=0)
            # Compute the average image by simply averaging the images
            img=[]
            if (self.params['dim'] is not None) and (self.results['pathnames'] is not None):
                img = np.vstack(list(map(lambda x: self.imread(x, colorscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=True), np.array(self.results['pathnames'])[idx])))
            eigen_img.append(imscale(np.mean(img, axis=0)))

            # dim = self._get_dim(eigen_img)
            # plt.figure();plt.imshow(eigen_img.reshape(dim))

            # Compute distance across all samples
            dist = distance.cdist(self.results['xycoord'], xycoord_center.reshape(-1, 1).T, metric=metric)
            # Take closest sample to the center
            idx_closest = np.argmin(dist)
            # Store
            center_idx.append(idx_closest)
            center_coord.append(xycoord_center)
            if self.results.get('pathnames', None) is not None:
                pathnames.append(self.results['pathnames'][idx_closest])
            else:
                pathnames.append('')

        # Store and return
        self.results_unique = {'labels': uilabels, 'idx': center_idx, 'xycoord_center': np.vstack(center_coord), 'pathnames': pathnames, 'img_mean': np.vstack(eigen_img)}
        return self.results_unique

    def find(self, Xnew, metric=None, k=None, alpha=0.05):
        """Find images that are similar to that of the input image.

        Finding images can be performed in two manners:

            * Based on the k-nearest neighbour
            * Based on significance after probability density fitting

        In both cases, the adjacency matrix is first computed using the distance metric (default Euclidean).
        In case of the k-nearest neighbour approach, the k nearest neighbours are determined.
        In case of significance, the adjacency matrix is used to to estimate the best fit for the loc/scale/arg parameters across various theoretical distribution.
        The tested disributions are *['norm', 'expon', 'uniform', 'gamma', 't']*. The fitted distribution is basically the similarity-distribution of samples.
        For each new (unseen) input image, the probability of similarity is computed across all images, and the images are returned that are P <= *alpha* in the lower bound of the distribution.
        If case both *k* and *alpha* are specified, the union of detected samples is taken.
        Note that the metric can be changed in this function but this may lead to confusions as the results will not intuitively match with the scatter plots as these are determined using metric in the fit_transform() function.

        Parameters
        ----------
        pathnames : list of str.
            Full path to images that are used in the model.
        metric : str, (default: the input of fit_transform()).
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'cityblock'
                * 'correlation'
                * 'cosine'
                * 'jaccard'
                * 'mahalanobis'
                * 'seuclidean'
                * 'sqeuclidean'
        k : int, (default: None)
            The k-nearest neighbour.
        alpha : float, default: 0.05
            Significance alpha.

        Returns
        -------
        dict containing keys with each input image that contains the following results.
            y_idx : list.
                Index of the detected/predicted images.
            distance : list.
                Absolute distance to the input image.
            y_proba : list
                Probability of similarity to the input image.
            y_filenames : list.
                filename of the detected image.
            y_pathnames : list.
                Pathname to the detected image.
            x_pathnames : list.
                Pathname to the input image.

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage(method='pca')
        >>>
        >>> # load example with faces
        >>> X, y = cl.import_example(data='mnist')
        >>>
        >>> # Cluster digits
        >>> results = cl.fit_transform(X)
        >>>
        >>> # Find images
        >>> results_find = cl.find(X[0,:], k=None, alpha=0.05)
        >>> cl.plot_find()
        >>> cl.scatter(zoom=3)
        >>>

        """
        out = None
        # Set the find_func to true to make sure the correct routines are used.
        self.find_func=True
        if (k is None) and (alpha is None):
            raise Exception(logger.error('Nothing to collect! input parameter "k" and "alpha" can not be None at the same time.'))
        if metric is None: metric=self.params_clusteval['metric']
        logger.info('Find similar images with metric [%s], k-nearest neighbors: %s and under alpha: %s ' %(metric, str(k), str(alpha)))

        # Check whether in is dir, list of files or array-like
        Xnew = self.import_data(Xnew)
        # Compute distance
        Y, feat = self._compute_distances(Xnew, metric=metric, alpha=alpha)
        # Collect the samples
        out = self._collect_pca(Xnew, Y, k, alpha, feat, todf=True)

        # Set the find_func to false because it is the end of this find function.
        self.find_func = False

        # Store
        self.results['predict'] = out
        self.params['metric_find'] = metric
        return self.results['predict']

    def extract_faces(self, pathnames):
        """Detect and extract faces from images.

        To cluster faces on images, we need to detect, and extract the faces from the images which is done in this function.
        Faces and eyes are detected using ``haarcascade_frontalface_default.xml`` and ``haarcascade_eye.xml`` in ``python-opencv``.

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
            pathnames_face : list of str.
                Filename of the extracted faces that are stored to disk.
            img : array-like.
                NxMxC for which N are the Samples, M the features and C the number of channels.
            coord_faces : array-like.
                list of lists containing coordinates fo the faces in the original image.
            coord_eyes : array-like.
                list of lists containing coordinates fo the eyes in the extracted (img and pathnames_face) image.

        Example
        -------
        >>> from clustimage import Clustimage
        >>>
        >>> # Init with default settings
        >>> cl = Clustimage(method='pca', grayscale=True)
        >>>
        >>> # Detect faces
        >>> face_results = cl.extract_faces(r'c://temp//my_photos//')
        >>> pathnames_face = face_results['pathnames_face']
        >>>
        >>> # Plot facces
        >>> cl.plot_faces(faces=True, eyes=True)
        >>>
        >>> # load example with faces
        >>> pathnames_face, y = cl.import_example(data='faces')
        >>>
        >>> # Cluster the faces
        >>> results = cl.fit_transform(pathnames_face)
        >>>
        >>>

        """
        # If face detection, grayscale should be True.
        if (not self.params['grayscale']): logger.warning('It is advisable to set "grayscale=True" when detecting faces.')

        # Read and pre-proces the input images
        X = self.import_data(pathnames)
        # Create empty list
        faces = {'img': [], 'pathnames': [], 'filenames': [], 'pathnames_face': [], 'coord_faces': [], 'coord_eyes': []}

        # Set logger to warning-error only
        verbose = logger.getEffectiveLevel()
        set_logger(verbose=30)

        # Extract faces and eyes from image
        for pathname in tqdm(X['pathnames'], disable=disable_tqdm(), desc='[clustimage]'):
            # Extract faces
            pathnames_face, imgfaces, coord_faces, coord_eyes, filename, path_to_image = self._extract_faces(pathname)
            # Store
            faces['pathnames'].append(path_to_image)
            faces['filenames'].append(filename)
            faces['pathnames_face'].append(pathnames_face)
            faces['img'].append(imgfaces)
            faces['coord_faces'].append(coord_faces)
            faces['coord_eyes'].append(coord_eyes)

        # Restore verbose status
        set_logger(verbose=verbose)
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
        # Make list of str
        if isinstance(pathnames, str): pathnames=[pathnames]

        # Filter images on min-number of pixels in image
        min_nr_pixels = 8

        # Check file existence on disk
        pathnames = np.array(pathnames)
        pathnames = pathnames[pathnames!=None]
        pathnames = list(pathnames[list(map(os.path.isfile, pathnames))])
        filenames = list(map(basename, pathnames))
        idx = range(0, len(pathnames))

        # Output dict
        out = {'img': None, 'pathnames': pathnames, 'filenames': filenames}

        # No need to import and process data when using hash function but we do not to check the image size and readability.
        logger.info("Reading and checking images.")
        if (self.params['method'] is not None) and ('hash' in self.params['method']):
            # logger.warning("In case of method=%s, flatten is set to False." %(self.params['method']))
            flatten=False

        # Read and preprocess data
        imgs = list(map(lambda x: self.imread(x, colorscale=grayscale, dim=dim, flatten=flatten, return_succes=True), tqdm(pathnames, disable=disable_tqdm(), desc='[clustimage]')))
        img, imgOK = zip(*imgs)
        img = np.array(img)

        # Remove the images that could not be read (and thus are False)
        I_corrupt = ~np.array(imgOK)
        if np.any(I_corrupt)>0:
            logger.info("[%.0d] Corrupt image(s) removed.", (sum(I_corrupt)))
            filenames = np.array(filenames)[~I_corrupt]
            pathnames = np.array(pathnames)[~I_corrupt]
            img = img[~I_corrupt]

        # Remove the images that are too small
        if np.where(np.array(list(map(len, img)))<min_nr_pixels)[0]:
            logger.info("Images with < %.0d pixels are detected and excluded.", (min_nr_pixels))

        idx = np.where(np.array(list(map(len, img)))>=min_nr_pixels)[0]
        img = img[idx]
        try:
            if flatten: img = np.vstack(img)
        except:
            logger.error("Images have different sizes, set the 'dim' parameter during initialization to scale all images to the same size. Hint: cl = Clustimage(dim=(128, 128))")

        # Remove not readable images and return
        out['filenames'] = np.array(filenames)[idx]
        out['pathnames'] = np.array(pathnames)[idx]
        out['img'] = img
        return out

    def extract_hog(self, X, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), flatten=True):
        """Extract HOG features.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.

        Returns
        -------
        feat : array-like
            NxF array for which N are the samples and F the reduced feature space.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from clustimage import Clustimage
        >>>
        >>> # Init
        >>> cl = Clustimage(method='hog')
        >>>
        >>> # Load example data
        >>> pathnames = cl.import_example(data='flowers')
        >>> # Read image according the preprocessing steps
        >>> img = cl.imread(pathnames[0], dim=(128,128))
        >>>
        >>> # Extract HOG features
        >>> img_hog = cl.extract_hog(img)
        >>>
        >>> plt.figure();
        >>> fig,axs=plt.subplots(1,2)
        >>> axs[0].imshow(img.reshape(128,128,3))
        >>> axs[0].axis('off')
        >>> axs[0].set_title('Preprocessed image', fontsize=10)
        >>> axs[1].imshow(img_hog.reshape(128,128), cmap='binary')
        >>> axs[1].axis('off')
        >>> axs[1].set_title('HOG', fontsize=10)

        """
        # Must be flattend array
        # if len(X.shape)>1:
        # raise Exception(logger.error('Input must be flattend grayscale image. Hint: During init set "grayscale=True" or imread(colorscale=0, flatten=True)'))
        # If 1D-vector, make 2D-array
        if len(X.shape)==1:
            X = X.reshape(-1, 1).T

        # Set dim correctly for reshaping image
        dim = self.get_dim(X)

        # Reshape data
        # if len(X.shape)==1:
        # X = X.reshape(dim)

        # Extract hog features per image
        if flatten:
            feat = list(map(lambda x: hog(x.reshape(dim), orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)[1].flatten(), tqdm(X, disable=disable_tqdm(), desc='[clustimage]')))
            # Stack all hog features in NxM array
            feat = np.vstack(feat)
        else:
            feat = list(map(lambda x: hog(x.reshape(dim), orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)[1], tqdm(X, disable=disable_tqdm(), desc='[clustimage]')))[0]
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
        self.pca = pca(**self.params_pca, verbose='warning')
        self.pca.fit_transform(X['img'], row_labels=X['filenames'])
        # Return
        return self.pca.results['PC'].values

    def import_data(self, Xraw, flatten=True, black_list=None):
        """Import images and return in an consistent manner.

        The input for the import_data() can have multiple forms; path to directory, list of strings and and array-like input.
        This requires that each of the input needs to be processed in its own manner but each should return the same structure to make it compatible across all functions.
        The following steps are used for the import:
            1. Images are imported with specific extention (['png','tiff','jpg']).
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
        flatten : Bool, (default: True)
            Flatten the processed NxMxC array to a 1D-vector
        black_list : list, (default: None)
            Exclude directory with all subdirectories from processing.

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
            Xraw = listdir(Xraw, ext=self.params['ext'], black_list=black_list)
            logger.info('[%s] images are extracted.', len(Xraw))

        logger.info("Reading and checking images.")

        # Return if no images are extracted.
        if len(Xraw)==0:
            return {'img': Xraw, 'pathnames': None, 'filenames': None}

        # Check string
        if isinstance(Xraw, str) and os.path.isfile(Xraw):
            Xraw = [Xraw]

        filenames, pathnames = None, None
        # Check pandas dataframe
        if isinstance(Xraw, pd.DataFrame):
            filenames = Xraw.index.values
            if len(np.unique(filenames))!=Xraw.shape[0]: raise Exception(logger.error("Filenames must be unique."))
            Xraw = Xraw.values

        # 2. Read images
        if isinstance(Xraw, dict):
            logger.info('Skipping import because import is already performed outside the fit_transform()')
            self.results = Xraw
        elif isinstance(Xraw, list) or isinstance(Xraw[0], str):
            # Make sure that list in lists are flattend
            Xraw = np.hstack(Xraw)
            # Check whether url and store all url images to tempdir on disk.
            Xraw = url2disk(Xraw, self.params['tempdir'])
            # Do not store in the object if the find functionality is used
            X = self.preprocessing(Xraw['pathnames'], grayscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=flatten)
            # Add the url location
            if Xraw['url'] is not None:
                IA, IB = ismember(X['pathnames'], Xraw['pathnames'].astype(str))
                X['url'] = np.repeat(None, len(X['pathnames']))
                X['url'][IA] = Xraw['url'][IB]

            if self.find_func:
                return X
            else:
                defaults = self.results
                # Read images and preprocessing
                self.results = X
                # Add remaining output variables
                self.results = {**defaults, **self.results}
        # 3. If input is array-like. Make sure X becomes compatible.
        elif isinstance(Xraw, np.ndarray):
            # Make 2D
            if len(Xraw.shape)==1:
                Xraw = Xraw.reshape(-1, 1).T
            # Check dim
            self.params['dim'] = self.get_dim(Xraw)
            # Scale the image
            logger.info('Scaling images..')
            Xraw = np.vstack(list(map(lambda x: imscale(x), Xraw)))
            # Store to disk
            if self.params['store_to_disk']:
                pathnames, filenames = store_to_disk(Xraw, self.params['dim'], self.params['tempdir'], files=filenames)
                pathnames = np.array(pathnames)
                filenames = np.array(filenames)

            # Make dict
            if self.find_func:
                return {'img': Xraw, 'pathnames': pathnames, 'filenames': filenames}
            else:
                self.results['img'] = Xraw
                self.results['pathnames'] = pathnames
                self.results['filenames'] = filenames

        # Return
        return self.results

    def clean_init(self):
        """Clean or removing previous results and models to ensure correct working."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if hasattr(self, 'results'): del self.results
            # if hasattr(self, 'results_faces'): del self.results_faces
            if hasattr(self, 'results_unique'): del self.results_unique
            if hasattr(self, 'distfit'): del self.distfit
            if hasattr(self, 'clusteval'): del self.clusteval
            if hasattr(self, 'pca'): del self.pca
        # Store results
        self.results = {'img': None, 'feat': None, 'xycoord': None, 'pathnames': None, 'labels': None, 'url': None}

    def embedding(self, X, metric="euclidean", embedding=None):
        """Compute embedding for the extracted features.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'cityblock'
                * 'correlation'
                * 'cosine'
                * 'jaccard'
                * 'mahalanobis'
                * 'seuclidean'
                * 'sqeuclidean'
        embedding : str, (default: retrieve from init)
            Perform embedding on the extracted features. The xycoordinates are used for plotting purposes.
            For UMAP parameters set set to default with densmap=True.
                * 'tsne'
                * 'umap'
                * None: Return the first to axis of input data X.

        Returns
        -------
        xycoord : array-like.
            x,y coordinates after embedding or alternatively the first 2 features.
        """
        if X.shape[0]<=2:
            return [0, 0]
        if embedding is None:
            embedding = self.params.get('embedding', 'tsne')

        logger.info('Compute [%s] embedding', self.params['embedding'])
        # Embedding
        if self.params['embedding']=='tsne':
            perplexity = np.minimum(X.shape[0] - 1, 30)
            xycoord = TSNE(n_components=2, init='random', metric=metric, perplexity=perplexity).fit_transform(X)
        elif self.params['embedding']=='umap':
            logger.info('Due to a "numba" error, UMAP is temporarily disabled.')
            # um = UMAP(densmap=True)
            # xycoord = um.fit_transform(X)
            xycoord = X[:, 0:2]
        else:
            xycoord = X[:, 0:2]
        # Return
        self.results['xycoord'] = xycoord
        # Return
        return self.results['xycoord']

    def extract_feat(self, Xraw):
        """Extract features based on the input data X.

        Parameters
        ----------
        Xraw : dict containing keys:
            img : array-like.
            pathnames : list of str.
            filenames : list of str.

        Returns
        -------
        X : array-like
            Extracted features.

        """
        # If the input is a directory, first collect the images from path
        logger.info('Extracting features using method: [%s]', self.params['method'])
        # Extract features
        if self.params['method']=='pca':
            Xfeat = self.extract_pca(Xraw)
        elif self.params['method']=='hog':
            Xfeat = self.extract_hog(Xraw['img'], orientations=self.params_hog['orientations'], pixels_per_cell=self.params_hog['pixels_per_cell'], cells_per_block=self.params_hog['cells_per_block'])
        elif self.params['method']=='pca-hog':
            Xfeat = {}
            Xfeat['img'] = self.extract_hog(Xraw['img'], orientations=self.params_hog['orientations'], pixels_per_cell=self.params_hog['pixels_per_cell'], cells_per_block=self.params_hog['cells_per_block'])
            Xfeat['filenames'] = Xraw['filenames']
            Xfeat = self.extract_pca(Xfeat)
        elif (self.params['method'] is not None) and ('hash' in self.params['method']):
            # Compute hash
            # hashs = list(map(self.compute_hash, tqdm(Xraw['pathnames'], disable=disable_tqdm())))
            Xfeat = list(map(self.compute_hash, tqdm(Xraw['img'], disable=disable_tqdm(), desc='[clustimage]')))
            Xfeat = np.array(Xfeat)

            # Removing hashes from images that could not be read
            # idx = np.where(np.array(list(map(len, hashs)))>1)[0]
            # Xraw['pathnames'] = np.array(Xraw['pathnames'])[idx]
            # hashs=np.array(hashs)[idx]

            # Build adjacency matrix with hash differences
            # logger.info('Build adjacency matrix [%gx%g] with %s differences.' %(len(hashs), len(hashs), self.params['method']))
            # Xfeat = np.abs(np.subtract.outer(hashs, hashs)).astype(float)
            # hex(int(''.join(hashs[0].hash.ravel().astype(int).astype(str)), 2))
            # plt.hist(diff[np.triu_indices(diff.shape[0], k=1)], bins=50)
        else:
            # Read images and preprocessing and flattening of images
            Xfeat = Xraw['img'].copy()

        # Store results
        if not self.find_func:
            self.results['feat'] = Xfeat
        # Message
        logger.info("Extracted features using [%s]: samples=%g, features=%g" %(self.params['method'], Xfeat.shape[0], Xfeat.shape[1]))
        # Return
        return Xfeat

    def compute_hash(self, img, hash_size=None):
        """Compute hash.

        Parameters
        ----------
        img : numpy-array
            Image.

        Returns
        -------
        imghash : numpy-array
            Hash.

        """
        imghash=[]
        if hash_size is None: hash_size=self.params_hash['hash_size']
        try:
            if self.params['method']=='crop-resistant-hash':
                try:
                    imghash = self.params_hash['hashfunc'](Image.fromarray(img)).segment_hashes[1].hash.ravel()
                except:
                    imghash = self.params_hash['hashfunc'](Image.fromarray(img)).segment_hashes[0].hash.ravel()
                self.params_hash['hash_size']=np.sqrt(len(imghash))
            else:
                imghash = self.params_hash['hashfunc'](Image.fromarray(img), hash_size=hash_size).hash.ravel()
        except:
            logger.error('Could not compute hash for a particular image.')
            pass

        return imghash

    def _compute_distances(self, X, metric, alpha):
        """Compute distances and probabilities for new unseen samples.

        In case of PCA, a transformation needs to take place first.

        Parameters
        ----------
        X : array-like
            NxM array for which N are the samples and M the features.
        metric : str, (default: 'euclidean').
            Distance measures. All metrics from sklearn can be used such as:
                * 'euclidean'
                * 'hamming'
                * 'cityblock'
                * 'correlation'
                * 'cosine'
                * 'jaccard'
                * 'mahalanobis'
                * 'seuclidean'
                * 'sqeuclidean'
        alpha : float, default: 0.05
            Significance alpha.

        """
        if self.params['method']=='pca':
            # Transform new unseen datapoint into feature space
            Xmapped = self.pca.transform(X['img'], row_labels=X['filenames'])
        elif self.params['method']=='pca-hog':
            # Extract Features
            X_feat = self.extract_hog(X['img'], orientations=self.params_hog['orientations'], pixels_per_cell=self.params_hog['pixels_per_cell'], cells_per_block=self.params_hog['cells_per_block'])
            # Transform new unseen datapoint into feature space
            Xmapped = self.pca.transform(X_feat, row_labels=X['filenames'])
            # Compute distance from input sample to all other samples
            # Y = distance.cdist(self.results['feat'].T, Xmapped, metric=metric)
        else:
            # Extract Features
            Xmapped = self.extract_feat(X)
            # Compute distance from input sample to all other samples
        Y = distance.cdist(self.results['feat'], Xmapped, metric=metric)

        # Sanity check
        if np.any(np.isnan(Y)):
            logger.warning('The metric [%s] results in NaN! Please change metric for appropriate results!', metric)

        # Fit distribution to emperical data and compute probability of the distances of interest
        if (alpha is not None) and ((not hasattr(self, 'distfit')) or (self.distfit is None) or (self.params['metric_find'] != metric)):
            # Compute distance across all samples
            Ytot = distance.cdist(self.results['feat'], self.results['feat'], metric=metric)
            # Take a subset of samples to prevent high computation times.
            x_max, y_max = np.minimum(500, Ytot.shape[0]), np.minimum(500, Ytot.shape[1])
            xrow, yrow = random.sample(range(1, x_max), x_max -1), random.sample(range(1, y_max), y_max -1)
            Ytot = Ytot[xrow, :]
            Ytot = Ytot[:, yrow]
            # Init distfit
            self.distfit = distfit(bound='down', multtest=None, distr=['norm', 'expon', 'uniform', 'gamma', 't'], verbose='warning')
            # Fit theoretical distribution
            _ = self.distfit.fit_transform(Ytot)
            # self.distfit.plot()
        else:
            logger.info('Loading pre-fitted theoretical model..')

        # Sanity check
        if len(X['filenames'])!=Y.shape[1]: raise Exception(logger.error('Number of input files does not match number of computed distances.'))
        # Return
        return Y, Xmapped

    def _extract_faces(self, pathname):
        """Extract the faces from the image.

        Parameters
        ----------
        pathname : str.
            Full path to a single image.

        Returns
        -------
        pathnames_face : list of str.
            Filename of the extracted faces that are stored to disk.
        img : array-like.
            NxMxC for which N are the Samples, M the features and C the number of channels.
        pathnames : list of str.
            Full path to images that are used in the model.
        coord_faces : array-like.
            list of lists containing coordinates fo the faces in the original image.
        coord_eyes : array-like.
            list of lists containing coordinates fo the eyes in the extracted (img and pathnames_face) image.
        filenames : list of str.
            Filename of the input images.
        pathnames : list of str.
            Pathnames of the input images.

        """
        # Set defaults
        coord_eyes, pathnames_face, imgstore = [], [], []
        # Get image
        X = self.preprocessing(pathname, grayscale=self.params['cv2_imread_colorscale'], dim=None, flatten=False)
        # Get the image and Convert into grayscale if required
        img = X['img'][0]

        # Detect faces using the face_cascade
        face_cascade = eval(self.params['face_cascade'])
        coord_faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # coord_faces = self.params['face_cascade'].detectMultiScale(img, 1.3, 5)
        # Setup the eye cascade
        eye_cascade = eval(self.params['eye_cascade'])

        # Collect the faces from the image
        for (x, y, w, h) in coord_faces:
            # Create filename for face
            filename = os.path.join(self.params['dirpath'], str(uuid.uuid4())) +'.png'
            pathnames_face.append(filename)
            # Store faces seperately
            imgface = imresize(img[y:y +h, x:x +w], dim=self.params['dim_face'])
            # Write to disk
            cv2.imwrite(filename, imgface)
            # Store face image
            # imgstore.append(imgface.flatten())
            imgstore.append(img_flatten(imgface))
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(imgface)
            # eyes = self.params['eye_cascade'].detectMultiScale(imgface)
            if eyes==(): eyes=None
            coord_eyes.append(eyes)
        # Return
        return pathnames_face, np.array(imgstore), coord_faces, coord_eyes, X['filenames'][0], X['pathnames'][0]

    def _collect_pca(self, X, Y, k, alpha, feat, todf=True):
        """Collect the samples that are closest in according the metric."""
        filenames = X['filenames']
        out = {}
        out['feat'] = feat

        # Collect nearest neighor and sample with highes probability per input sample
        for i, filename in enumerate(filenames):
            store_key = {}
            idx_dist, idx_k = None, None
            # Collect bes samples based on k-nearest neighbor
            if k is not None:
                idx_k = np.argsort(Y[:, i])[0:k]
            # Collect samples based on probability
            if alpha is not None:
                dist_results = self.distfit.predict(Y[:, i], verbose=0)
                idx_dist = np.where(dist_results['y_proba']<=alpha)[0]
                # Sort on significance
                idx_dist = idx_dist[np.argsort(dist_results['y_proba'][idx_dist])]
            else:
                # If alpha is not used, set all to nan
                dist_results={}
                dist_results['y_proba'] = np.array([np.nan] *Y.shape[0])

            # Combine the unique k-nearest samples and probabilities.
            idx = unique_no_sort(np.append(idx_dist, idx_k))
            # Store in dict
            logger.info('[%d] similar images found for [%s]' %(len(idx), filename))
            store_key = {**store_key, 'y_idx': idx, 'distance': Y[idx, i], 'y_proba': dist_results['y_proba'][idx], 'labels': np.array(self.results['labels'])[idx].tolist(), 'y_filenames': np.array(self.results['filenames'])[idx].tolist(), 'y_pathnames': np.array(self.results['pathnames'])[idx].tolist(), 'x_pathnames': X['pathnames'][i]}
            if todf: store_key = pd.DataFrame(store_key)
            out[filename] = store_key

        # Return
        return out

    def imread(self, filepath, colorscale=1, dim=(128, 128), flatten=True, return_succes=False):
        """Read and pre-processing of images.

        The pre-processing has 4 steps and are exectued in this order.
            * 1. Import data.
            * 2. Conversion to gray-scale (user defined)
            * 3. Scaling color pixels between [0-255]
            * 4. Resizing

        Parameters
        ----------
        filepath : str
            Full path to the image that needs to be imported.
        colorscale : int, default: 1 (gray)
            colour-scaling from opencv.
            * 0: cv2.IMREAD_GRAYSCALE
            * 1: cv2.IMREAD_COLOR
            * 2: cv2.IMREAD_ANYDEPTH
            * 8: cv2.COLOR_GRAY2RGB
            * -1: cv2.IMREAD_UNCHANGED
        dim : tuple, (default: (128,128))
            Rescale images. This is required because the feature-space need to be the same across samples.
        flatten : Bool, (default: True)
            Flatten the processed NxMxC array to a 1D-vector

        Returns
        -------
        img : array-like
            Imported and processed image.

        Examples
        --------
        >>> # Import libraries
        >>> from clustimage import Clustimage
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Init
        >>> cl = Clustimage()
        >>>
        >>> # Load example dataset
        >>> pathnames = cl.import_example(data='flowers')
        >>> # Preprocessing of the first image
        >>> img = cl.imread(pathnames[0], dim=(128,128), colorscale=1)
        >>>
        >>> # Plot
        >>> fig, axs = plt.subplots(1,2, figsize=(15,10))
        >>> axs[0].imshow(cv2.imread(pathnames[0])); plt.axis('off')
        >>> axs[1].imshow(img.reshape(128,128,3)); plt.axis('off')
        >>> fig
        >>>

        """
        # logger.debug('[%s]' %(filepath))
        img=[]
        readOK = False
        try:
            # Read the image
            img = _imread(filepath, colorscale=colorscale)
            # Scale the image
            img = imscale(img)
            # Resize the image
            img = imresize(img, dim=dim)
            # Flatten the image
            if flatten: img = img_flatten(img)
            # OK
            readOK = True
        except:
            logger.warning('Could not read: [%s]' %(filepath))
        # Return
        if return_succes:
            return img, readOK
        else:
            return img

    def save(self, filepath='clustimage.pkl', overwrite=False):
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
        if (filepath is None) or (filepath=='') or (filepath=='clustimage.pkl'):
            filepath = self.params['filepath']
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'
        self.params['filepath'] = filepath

        # Store data
        storedata = {}
        storedata['results'] = self.results
        storedata['params'] = self.params
        if hasattr(self, 'pca'): self.params_pca['model'] = self.pca
        storedata['params_pca'] = self.params_pca
        storedata['params_hog'] = self.params_hog
        storedata['params_hash'] = self.params_hash
        if hasattr(self, 'results_faces'): storedata['results_faces'] = self.results_faces
        if hasattr(self, 'results_unique'): storedata['results_unique'] = self.results_unique
        if hasattr(self, 'distfit'): storedata['distfit'] = self.distfit
        if hasattr(self, 'clusteval'): storedata['clusteval'] = self.clusteval

        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=3)
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
        if (filepath is None) or (filepath=='') or (filepath=='clustimage.pkl'):
            filepath = self.params['filepath']
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'

        # Load
        storedata = pypickle.load(filepath, verbose=verbose)

        # Restore the data in self
        if storedata is not None:
            self.results = storedata['results']
            self.params = storedata['params']
            self.params_pca = storedata['params_pca']
            self.pca = storedata['params_pca'].get('model', None)
            self.params_hog = storedata['params_hog']
            self.params_hash = storedata['params_hash']
            self.results_faces = storedata.get('results_faces', None)
            self.results_unique = storedata.get('results_unique', None)
            self.distfit = storedata.get('distfit', None)
            self.clusteval = storedata.get('clusteval', None)
            self.find_func = False

            logger.info('Load succesful!')
            # Return results
            # return self.results
        else:
            logger.warning('Could not load previous results!')

    def plot_faces(self, faces=True, eyes=True, cmap=None):
        """Plot detected faces.

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
        cmap = _set_cmap(cmap, self.params['grayscale'])
        # Set logger to warning-error only
        verbose = logger.getEffectiveLevel()
        set_logger(verbose=30)

        # Walk over all detected faces
        if hasattr(self, 'results_faces'):
            for i, pathname in tqdm(enumerate(self.results_faces['pathnames']), disable=disable_tqdm(), desc='[clustimage]'):
                # Import image
                img = self.preprocessing(pathname, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()

                # Plot the faces
                if faces:
                    coord_faces = self.results_faces['coord_faces'][i]
                    plt.figure()
                    for (x, y, w, h) in coord_faces:
                        cv2.rectangle(img, (x, y), (x +w, y +h), (255, 0, 0), 2)
                    # if len(img.shape)==3:
                    #     plt.imshow(img[:, :, ::-1], cmap=cmap)  # RGB-> BGR
                    # else:
                        plt.imshow(img, cmap=cmap)

                # Plot the eyes
                if eyes:
                    coord_eyes = self.results_faces['coord_eyes'][i]
                    for k in np.arange(0, len(self.results_faces['pathnames_face'][i])):
                        # face = self.results_faces['img'][i][k].copy()
                        pathnames_face = self.results_faces['pathnames_face'][i][k]
                        if os.path.isfile(pathnames_face):
                            face = self.preprocessing(pathnames_face, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()
                            if coord_eyes[k] is not None:
                                plt.figure()
                                for (ex, ey, ew, eh) in coord_eyes[k]:
                                    cv2.rectangle(face, (ex, ey), (ex +ew, ey +eh), (0, 255, 0), 2)
                                if len(face.shape)==3:
                                    plt.imshow(face[:, :, ::-1])  # RGB-> BGR
                                else:
                                    plt.imshow(face)
                        else:
                            logger.warning('File is removed: %s', pathnames_face)

                # Pause to plot to screen
                plt.pause(0.1)
        else:
            logger.warning('Nothing to plot. First detect faces with ".extract_faces(pathnames)"')

        set_logger(verbose=verbose)

    def dendrogram(self, max_d=None, figsize=(15, 10), update_labels=True):
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
        results = None
        self._check_status()

        if hasattr(self, 'clusteval'):
            results = self.clusteval.dendrogram(max_d=max_d, figsize=figsize)
            if results.get('labx', None) is not None:
                results['labels'] = results['labx']
                results.pop('labx')
            if update_labels:
                self.results['labels'] = results['labels']
                self.results_unique = self.unique()
                logger.info('Updating cluster-labels ')
        else:
            logger.warning('This Plot requires running fit_transform()')

        return results

    def _add_img_to_scatter(self, ax, pathnames, xycoord, cmap=None, zoom=0.2):
        # Plot the images on top of the scatterplot
        if zoom is not None and self.params['store_to_disk']:
            for i, pathname in enumerate(pathnames):
                if isinstance(pathname, str):
                    img = self.imread(pathname, dim=self.params['dim'], colorscale=self.params['cv2_imread_colorscale'], flatten=False)
                else:
                    dim = self.get_dim(pathname)
                    # dim = self.get_dim(pathname)
                    # plt.figure();plt.imshow(eigen_img.reshape(dim))
                    img = pathname.reshape(dim)
                # Make hte plot
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, cmap=cmap, zoom=zoom), xycoord[i, :])
                ax.add_artist(imagebox)

    def scatter(self,
                dotsize=15,
                legend=False,
                zoom=0.3,
                img_mean=True,
                text=True,
                plt_all=False,
                density=False,
                figsize=(15, 10),
                ax=None,
                args_scatter={}):
        """Plot the samples using a scatterplot.

        Parameters
        ----------
        plt_all : bool, (default: False)
            False: Only plot the controid images.
            True: Plot all images on top of the scatter.
        dotsize : int, (default: 15)
            Dot size of the scatterpoints.
        legend : bool, (default: False)
            Plot the legend.
        zoom : bool, (default: 0.3)
            Plot the image in the scatterplot.
            None : Do not plot the image.
        text : bool, (default: True)
            Plot the cluster labels.
        density : bool, (default: Fale)
            Plot the density over the clusters.
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).
        args_scatter : dict, default: {}.
            Arguments for the scatter plot. The following are default:
            {'title': '',
            'fontsize': 18,
            'fontcolor': [0, 0, 0],
            'xlabel': 'x-axis',
            'ylabel': 'y-axis',
            'cmap': cmap,
            'density': False,
            'gradient': None,
            }

        Returns
        -------
        tuple (fig, ax)

        Examples
        --------
        >>> # Import library
        >>> from clustimage import Clustimage
        >>>
        >>> # Initialize with default settings.
        >>> cl = Clustimage()
        >>>
        >>> # Import example dataset
        >>> X, y = cl.import_example(data='mnist')
        >>>
        >>> # Run the model to find the optimal clusters.
        >>> results = cl.fit_transform(X)
        >>>
        >>> # Make scatter plots
        >>> cl.scatter()
        >>>
        >>> # More input arguments for the scatterplot
        >>> cl.scatter(dotsize=35, args_scatter={'fontsize':24, 'density':'#FFFFFF', 'cmap':'Set2'})

        """
        # Check status
        fig = None
        self._check_status()
        if self.results['xycoord'] is None:
            logger.warning('Missing x,y coordinates in results dict. Hint: try to first run: cl.embedding(Xfeat)')
            return None
        # Set default settings
        # if args_scatter.get('cmap', None) is None:
        cmap = plt.cm.gray if self.params['grayscale'] else 'Set1'
        # Get the cluster labels
        labels = self.results.get('labels', None)
        if labels is None: labels=np.zeros_like(self.results['xycoord'][:, 0]).astype(int)

        # Make scatterplot
        if args_scatter.get('title', None) is None:
            if self.params.get('cluster_space', None) is None:
                title = (self.params['embedding'] + ' plot.')
                logger.warning('Run .fit_transform() or .cluster() to colour based on the samples on the cluster labels.')
            else:
                title = (self.params['embedding'] + ' plot. Samples are coloured on the cluster labels (%s dimensional).' %(self.params['cluster_space']))

        # Add colors
        colours = colourmap.fromlist(labels, cmap=cmap, verbose=get_logger())
        args_scatter['c'] = colours[0]

        # Defaults
        default_scatter = {'title': title, 'fontsize': 18, 'fontcolor': [0, 0, 0], 'xlabel': 'x-axis', 'ylabel': 'y-axis', 'cmap': 'Set1', 'density': False, 'gradient': None, 'figsize': figsize, 'legend': True, 's': dotsize}
        args_scatter = {**default_scatter, **args_scatter}

        # Set logger to warning-error only
        # verbose = logger.getEffectiveLevel()
        # set_logger(verbose=40)

        # Scatter
        fig, ax = scatterd(self.results['xycoord'][:, 0], self.results['xycoord'][:, 1], labels=labels, ax=ax, **args_scatter, verbose=get_logger())

        if hasattr(self, 'results_unique'):
            if img_mean:
                X = self.results_unique['img_mean']
            else:
                X = self.results_unique['pathnames']
            self._add_img_to_scatter(ax, cmap=cmap, zoom=zoom, pathnames=X, xycoord=self.results_unique['xycoord_center'])

        if plt_all:
            self._add_img_to_scatter(ax, cmap=cmap, zoom=zoom, pathnames=self.results['pathnames'], xycoord=self.results['xycoord'])
        plt.show()
        plt.draw()

        # Scatter the predicted cases
        if (self.results.get('predict', None) is not None):
            if self.params['method']=='pca':
                logger.info('Plotting predicted results..')

                # Scatter all points
                fig, ax = self.pca.scatter(y=labels, legend=legend, label=False, figsize=figsize, s=dotsize, ax=ax)

                # Create unique colors
                # colours = colourmap.fromlist(self.results['predict']['feat'].index)[1]
                for key in self.results['predict'].keys():
                    if self.results['predict'].get(key).get('y_idx', None) is not None:
                        # Color based on the closest cluster label
                        x = self.results['predict']['feat'].iloc[:, 0].loc[key]
                        if len(self.results['predict']['feat'])>=2:
                            y = self.results['predict']['feat'].iloc[:, 1].loc[key]
                        else:
                            y=0

                        # Color based on the most often seen cluster label
                        uiy, ycounts = np.unique(self.results['predict'][key]['labels'], return_counts=True)
                        y_predict = uiy[np.argmax(ycounts)]

                        # Scatter
                        color = colours[1].get(y_predict, [0, 0, 0])
                        ax.scatter(x, y, color=color, edgecolors=[0, 0, 0], marker='*', s=dotsize * 10)
                        if text: ax.text(x, y, key, color=color)
                plt.show()
                plt.draw()
            else:
                logger.info('Mapping predicted results is only possible when uing method="pca".')

        # Restore verbose status
        # set_logger(verbose=verbose)
        # Return
        return fig, ax

    def plot_unique(self, cmap=None, img_mean=True, show_hog=False, figsize=(15, 10)):
        """Plot unique images.

        Parameters
        ----------
        cmap : str, (default: None)
            Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        img_mean : bool, (default: False)
            Plot the image mean.
        show_hog : bool, (default: False)
            Plot the hog features.
        figsize : tuple, (default: (15, 10).
            Size of the figure (height, width).

        Returns
        -------
        None.

        """
        if hasattr(self, 'results_unique'):
            # Set logger to warning-error only
            verbose = logger.getEffectiveLevel()
            set_logger(verbose=40)
            # Defaults
            imgs, imgshog = [], []
            cmap = _set_cmap(cmap, self.params['grayscale'])
            txtlabels = list(map(lambda x: 'Cluster' +x, self.results_unique['labels'].astype(str)))

            # Collect the image data
            if img_mean:
                subtitle='(averaged per cluster)'
                imgs=self.results_unique['img_mean']
                if (self.params['method']=='hog'):
                    for img in imgs:
                        hogtmp = self.extract_hog(img, pixels_per_cell=self.params_hog['pixels_per_cell'], orientations=self.params_hog['orientations'], flatten=False)
                        imgshog.append(hogtmp)
            else:
                # Collect all samples
                subtitle='(most centroid image per cluster)'
                for i, file in enumerate(self.results_unique['pathnames']):
                    img = self.imread(file, colorscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=True)
                    imgs.append(img)
                    if show_hog and (self.params['method']=='hog'):
                        idx=self.results_unique['idx'][i]
                        hogtmp = exposure.rescale_intensity(self.results['feat'][idx, :].reshape(self.params['dim']), in_range=(0, 10))
                        imgshog.append(hogtmp)

            self._make_subplots(imgs, None, cmap, figsize, title='Unique images ' +subtitle, labels=txtlabels)

            if show_hog and (self.params['method']=='hog'):
                self._make_subplots(imgshog, None, 'binary', figsize, title='Unique HOG images ' +subtitle, labels=txtlabels)

            # Restore verbose status
            set_logger(verbose=verbose)
        else:
            logger.warning('Plotting unique images is not possible. Hint: Try first to run: cl.unique()')

    def plot_find(self, cmap=None, figsize=(15, 10)):
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
        cmap = _set_cmap(cmap, self.params['grayscale'])

        # Plot the images that are similar to each other.
        if self.results.get('predict', None) is not None:
            for key in self.results['predict'].keys():
                try:
                    if (self.results['predict'].get(key).get('y_idx', None) is not None):
                        # Collect images
                        input_img = self.results['predict'][key]['x_pathnames'][0]
                        find_img = self.results['predict'][key]['y_pathnames']
                        # Input label
                        if isinstance(input_img, str): input_img=[input_img]
                        if isinstance(find_img, str): find_img=[find_img]
                        # Input images
                        I_input = list(map(lambda x: self.imread(x, colorscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=False), input_img))
                        # Predicted label
                        I_find = list(map(lambda x: self.imread(x, colorscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=False), find_img))
                        # Combine input image with the detected images
                        imgs = I_input + I_find
                        input_txt = basename(self.results['predict'][key]['x_pathnames'][0])
                        # Make the labels for the subplots
                        if not np.isnan(self.results['predict'][key]['y_proba'][0]):
                            labels = ['Input'] + list(map(lambda x: 'P={:.3g}'.format(x), self.results['predict'][key]['y_proba']))
                        else:
                            labels = ['Input'] + list(map(lambda x: 'k=' +x, np.arange(1, len(I_find) +1).astype(str)))
                        title = 'Find similar images for [%s].' %(input_txt)
                        # Make the subplot
                        self._make_subplots(imgs, None, cmap, figsize, title=title, labels=labels)
                        logger.info('[%d] similar images detected for input image: [%s]' %(len(find_img), key))
                except:
                    pass
        else:
            logger.warning('No prediction results are found. Hint: Try to run the .find() functionality first.')

    def plot(self, labels=None, show_hog=False, ncols=None, cmap=None, min_clust=1, figsize=(15, 10)):
        """Plot the results.

        Parameters
        ----------
        labels : list, (default: None)
            Cluster label to plot. In case of None, all cluster labels are plotted.
        ncols : int, (default: None)
            Number of columns to use in the subplot. The number of rows are estimated based on the columns.
        Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        show_hog : bool, (default: False)
            Plot the hog features next to the input image.
        min_clust : int, (default: 1)
            Plots are created for clusters with > min_clust samples
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        # Do some checks and set defaults
        self._check_status()
        cmap = _set_cmap(cmap, self.params['grayscale'])

        # Plot the clustered images
        if (self.results.get('labels', None) is not None) and (self.results.get('pathnames', None) is not None):
            # Set logger to error only
            verbose = logger.getEffectiveLevel()
            set_logger(verbose=50)

            # Gather labels
            if labels is None: labels = self.results['labels']
            if not isinstance(labels, list): labels = [labels]
            # Unique labels
            uilabels = np.unique(labels)

            # Run over all labels.
            for label in tqdm(uilabels, disable=disable_tqdm(), desc='[clustimage]'):
                idx = np.where(self.results['labels']==label)[0]
                if len(idx)>min_clust:
                    # Collect the images
                    getfiles = np.array(self.results['pathnames'])[idx]
                    getfiles = getfiles[np.array(list(map(lambda x: os.path.isfile(x), getfiles)))]

                    if len(getfiles)>0:
                        # Get the images that cluster together
                        imgs = list(map(lambda x: self.imread(x, colorscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=False), getfiles))
                        # Make subplots
                        if ncols is None:
                            ncol=np.maximum(int(np.ceil(np.sqrt(len(imgs)))), 2)
                        else:
                            ncol=ncols
                        self._make_subplots(imgs, ncol, cmap, figsize, ("Images in cluster %s" %(str(label))))

                        # Make hog plots
                        if show_hog and (self.params['method']=='hog'):
                            hog_images = self.results['feat'][idx, :]
                            fig, axs = plt.subplots(len(imgs), 2, figsize=(15, 10), sharex=True, sharey=True)
                            ax = axs.ravel()
                            fignum=0
                            for i, img in enumerate(imgs):
                                hog_image_rescaled = exposure.rescale_intensity(hog_images[i, :].reshape(self.params['dim']), in_range=(0, 10))
                                ax[fignum].imshow(img[..., ::-1], cmap=cmap)
                                ax[fignum].axis('off')
                                ax[fignum +1].imshow(hog_image_rescaled, cmap=cmap)
                                ax[fignum +1].axis('off')
                                fignum=fignum +2

                            _ = fig.suptitle('Histogram of Oriented Gradients', fontsize=16)
                            plt.tight_layout()
                            plt.show()

                        # Restore verbose status
                        set_logger(verbose=verbose)
                else:
                    # logger.error('The cluster clabel [%s] does not exsist! Skipping!', label)
                    pass
        else:
            logger.warning('Plotting is not possible. Path locations can not be found (maybe deleted with the clean_files functions). Hint: try to set "store_to_disk=True" during initialization.')

    def _get_rows_cols(self, n, ncols=None):
        # Setup rows and columns
        if ncols is None: ncols=np.maximum(int(np.ceil(np.sqrt(n))), 2)
        nrows = int(np.ceil(n /ncols))
        return nrows, ncols

    def _make_subplots(self, imgs, ncols, cmap, figsize, title='', labels=None):
        """Make subplots."""
        # Get appropriate dimension
        if self.params['grayscale']:
            dim = self.params['dim']
            dimlen=4
        else:
            dim = np.append(self.params['dim'], 3)
            dimlen=3

        # Setup rows and columns
        nrows, ncols = self._get_rows_cols(len(imgs), ncols=ncols)
        # Make figure
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        if self.params['store_to_disk']:

            # Make the actual plots
            for i, ax in enumerate(axs.ravel()):
                if i<len(imgs):
                    img = imgs[i]
                    if len(img.shape)==1:
                        img = img.reshape((dim[0], dim[1], dimlen))
                        img = img[:, :, : 3]
                    if len(img.shape)==3:
                        # ax.imshow(img[:, :, ::-1], cmap=cmap)  # RGB-> BGR
                        ax.imshow(img, cmap=cmap)
                    else:
                        ax.imshow(img, cmap=cmap)
                    if labels is not None: ax.set_title(labels[i])
                ax.axis("off")
            _ = fig.suptitle(title, fontsize=16)

        # Small pause to build the plot
        plt.pause(0.1)
        # Return the rows and columns
        return nrows, ncols

    def clean_files(self, clean_tempdir=False):
        """Clean files.

        Returns
        -------
        None.

        """
        # Cleaning
        from pathlib import Path
        if hasattr(self, 'results_faces'):
            out = []
            for sublist in self.results_faces['pathnames_face']:
                out.extend(sublist)

            p = Path(out[0])
            dirpath = str(p.parent)

            if os.path.isdir(dirpath):
                logger.info('Removing directory with all content: %s', dirpath)
                shutil.rmtree(dirpath)
                self.results['pathnames'] = None
                self.results['filenames'] = None

        # Cleaning temp directory
        if os.path.isdir(self.params['tempdir']):
            logger.info('Removing temp directory %s', self.params['tempdir'])

            files_in_tempdir = os.listdir(self.params['tempdir'])
            _, idx = ismember(files_in_tempdir, self.results['filenames'])
            logger.info('Removing images in temp directory %s', self.params['tempdir'])
            for i in idx:
                logger.debug('remove: %s', self.results['pathnames'][i])
                os.remove(self.results['pathnames'][i])
                self.results['filenames'][i]=None
                self.results['pathnames'][i]=None

            if clean_tempdir:
                logger.info('Removing the entire temp directory %s', self.params['tempdir'])
                shutil.rmtree(self.params['tempdir'])
                self.results['filenames'] = None
                self.results['pathnames'] = None

    def get_dim(self, Xraw, dim=None):
        """Determine dimension for image vector.

        Parameters
        ----------
        Xraw : array-like float
            Image vector.
        dim : tuple (int, int)
            Dimension of the image.

        Returns
        -------
        None.

        """
        if dim is None: dim = self.params['dim']
        return _get_dim(Xraw, dim=dim)

    def import_example(self, data='flowers', url=None, sep=','):
        """Import example dataset from github source.

        Import one of the datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Images:
                * 'faces'
                * 'mnist'
            Files with images:
                * 'southern_nebula'
                * 'flowers'
                * 'scenes'
                * 'cat_and_dog'

        url : str
            url link to to dataset.

        Returns
        -------
        list of str
            list of str containing filepath to images.

        """
        return import_example(data=data, url=url, sep=sep, verbose=get_logger())


# %% Store images to disk
def _get_dim(Xraw, dim, grayscale=None):
    dimOK=False
    # Determine the dimension based on the length of the 1D-vector.
    # if len(Xraw.shape)>=2:
    # dim = Xraw.shape
    # dimOK=True

    if not dimOK:
        if len(Xraw.shape)==1:
            Xraw = Xraw.reshape(-1, 1).T
        # Compute dim based on vector length
        dimX = int(np.sqrt(Xraw.shape[1]))

        if (len(Xraw.shape)==1) and ((dimX!=dim[0]) or (dimX!=dim[1])):
            logger.warning('The default dim=%s of the image does not match with the input: %s. Set dim=%s during initialization!' %(str(dim), str([int(dimX)] *2), str([int(dimX)] *2)))

        try:
            Xraw[0, :].reshape(dim)
            dimOK=True
        except:
            pass

    if not dimOK:
        try:
            Xraw[0, :].reshape(np.append(dim, 3))
            dim=np.append(dim, 3)
            dimOK=True
        except:
            pass

    if not dimOK:
        try:
            Xraw[0, :].reshape([dimX, dimX])
            dim = [dimX, dimX]
            dimOK=True
        except:
            pass

    if not dimOK:
        try:
            Xraw[0, :].reshape([dimX, dimX, 3])
            dim = [dimX, dimX, 3]
        except:
            pass

    if not dimOK:
        raise Exception(logger.error('The default dim=%s of the image does not match with the input: %s. Set dim=%s during initialization!' %(str(dim), str([int(dimX)] *2), str([int(dimX)] *2))))
    else:
        logger.debug('The dim is changed into: %s', str(dim))

    return dim

# %% Store images to disk


def store_to_disk(Xraw, dim, tempdir, files=None):
    """Store to disk."""
    # Determine the dimension based on the length of the vector.
    dim = _get_dim(Xraw, dim)
    # Store images to disk
    filenames, pathnames = [], []

    logger.info('Writing images to tempdir [%s]', tempdir)
    for i in tqdm(np.arange(0, Xraw.shape[0]), disable=disable_tqdm(), desc='[clustimage]'):
        if files is not None:
            filename = files[i]
        else:
            filename = str(uuid.uuid4()) +'.png'

        pathname = os.path.join(tempdir, filename)
        # Write to disk
        img = imscale(Xraw[i, :].reshape(dim))
        cv2.imwrite(pathname, img)
        filenames.append(filename)
        pathnames.append(pathname)
    return pathnames, filenames


# %% Unique without sort
def unique_no_sort(x):
    """Uniques without sort."""
    x = x[x!=None]
    indexes = np.unique(x, return_index=True)[1]
    return [x[index] for index in sorted(indexes)]


# %% Extract basename from path
def basename(label):
    """Extract basename from path."""
    return os.path.basename(label)


# %% Resize image
def img_flatten(img):
    """Flatten image."""
    return img.flatten()


# %% Resize image
def imresize(img, dim=(128, 128)):
    """Resize image."""
    if (dim is not None) and (img.shape[0:2] != dim):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


# %% Set cmap
def _set_cmap(cmap, grayscale):
    """Set the colourmap."""
    if cmap is None:
        cmap = 'gray' if grayscale else None
    return cmap


# %% Scaling
def imscale(img):
    """Normalize image by scaling.

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
def _imread(filepath, colorscale=1):
    """Read image from filepath using colour-scaling.

    Parameters
    ----------
    filepath : str
        path to file.
    colorscale : int, default: 1 (gray)
        colour-scaling from opencv.
        * 0: cv2.IMREAD_GRAYSCALE
        * 1: cv2.IMREAD_COLOR
        * 8: cv2.COLOR_GRAY2RGB

    Returns
    -------
    img : numpy array
        raw rgb or gray image.

    """
    img=None
    # if os.path.isfile(filepath):
    # Read the image
    if colorscale==0:
        # In case of gray-scale:
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath)

    # plt.imshow(img)
    # Convert Image to numpy array
    # It's not the most efficient way, but it works.
    img = np.asarray(img)

    # Remove alpha channel if existent
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, : 3]

    # Restore to RGB colors
    # if colorscale==0:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # In case of rgb images: make gray images compatible with RGB
    if ((colorscale!=0) and (colorscale!=6)) and (len(img.shape)<3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # else:
    #     logger.warning('File does not exists: %s', filepath)
    # plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    return cv2.cvtColor(img, colorscale)

# %% Read image
# def _imread_old(filepath, colorscale=1):
#     """Read image from filepath using colour-scaling.

#     Parameters
#     ----------
#     filepath : str
#         path to file.
#     colorscale : int, default: 1 (gray)
#         colour-scaling from opencv.
#         * 0: cv2.IMREAD_GRAYSCALE
#         * 1: cv2.IMREAD_COLOR
#         * 8: cv2.COLOR_GRAY2RGB

#     Returns
#     -------
#     img : numpy array
#         raw rgb or gray image.

#     """
#     img=None
#     # if os.path.isfile(filepath):
#     # Read the image
#     img = cv2.imread(filepath, colorscale)

#     # In case of rgb images: make gray images compatible with RGB
#     if ((colorscale!=0) and (colorscale!=6)) and (len(img.shape)<3):
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     # else:
#     #     logger.warning('File does not exists: %s', filepath)

#     return img


# %%
def get_logger():
    """Return logger status."""
    return logger.getEffectiveLevel()


# %%
def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical']: Messages from critical level and higher.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'error': 50,
                  'critical': 50}
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


# %% import examples
def import_example(data='flowers', url=None, sep=',', verbose='info'):
    """Import example dataset from github source.

    Import the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Images:
            * 'faces'
            * 'mnist'
        Files with images:
            * 'southern_nebula'
            * 'flowers'
            * 'scenes'
            * 'cat_and_dog'

    url : str
        url link to to dataset.

    Returns
    -------
    list of str
        list of str containing filepath to images.

    Returns
    -------
    list or numpy array
    """
    # Dowload
    df = dz.get(data=data, url=url, sep=sep, verbose=verbose)
    # Proces
    if data=='mnist' or data=='faces':
        X=df.iloc[:, 1:].values
        y=df['target'].values
        return X, y
    else:
        return df


# %% Recursively list files from directory
def listdir(dirpath, ext=['png', 'tiff', 'jpg'], black_list=None):
    """Collect recursive images from path.

    Parameters
    ----------
    dirpath : str
        Path to directory; "/tmp" or "c://temp/"
    ext : list, default: ['png','tiff','jpg']
        extentions to collect form directories.
    black_list : list, (default: ['undouble'])
        Exclude directory with all subdirectories from processing.

    Returns
    -------
    getfiles : list of str.
        Full pathnames to images.

    Example
    -------
    >>> import clustimage as cl
    >>> pathnames = cl.listdir('c://temp//flower_images')

    """
    if isinstance(ext, str): ext = [ext]
    if not isinstance('dirpath', str): raise Exception(print('Error: "dirpath" should be of type string.'))
    if not os.path.isdir(dirpath): raise Exception(print('Error: The directory can not be found: %s.' %dirpath))

    getfiles = []
    for root, _, filenames in os.walk(dirpath):
        # Check if the (sub)directory is black listed
        bl_found = np.isin(os.path.split(root)[1], black_list)
        if (black_list is None) or (not bl_found):
            for iext in ext:
                for filename in fnmatch.filter(filenames, '*.' +iext):
                    getfiles.append(os.path.join(root, filename))
        else:
            logger.info('Excluded: <%s>' %(root))
    logger.info('[%s] files are collected recursively from path: [%s]', len(getfiles), dirpath)
    return getfiles


# %% Get image HASH function
def get_params_hash(hashmethod, params_hash={}):
    """Get image hash function.

    Parameters
    ----------
    hashmethod : str (default: 'ahash')
        'ahash': Average hash
        'phash': Perceptual hash
        'dhash': Difference hash
        'whash-haar': Haar wavelet hash
        'whash-db4': Daubechies wavelet hash
        'colorhash': HSV color hash
        'crop-resistant-hash': Crop-resistant hash

    Returns
    -------
    hashfunc : Object

    """
    if hashmethod == 'ahash':
        hashfunc = imagehash.average_hash
    elif hashmethod == 'phash':
        hashfunc = imagehash.phash
    elif hashmethod == 'dhash':
        hashfunc = imagehash.dhash
    elif hashmethod == 'whash-haar':
        hashfunc = imagehash.whash
    elif hashmethod == 'whash-db4':
        def hashfunc(img):
            return imagehash.whash(img, mode='db4')
    elif hashmethod == 'colorhash':
        hashfunc = imagehash.colorhash
    elif hashmethod == 'crop-resistant-hash':
        hashfunc = imagehash.crop_resistant_hash
    else:
        hashfunc=None
        hashmethod=None

    hash_defaults={'threshold': 0, 'hash_size': 8}
    # Set the hash parameters
    params_hash = {**hash_defaults, **params_hash}
    # self.params_hash = params_hash
    params_hash['hashfunc'] = hashfunc
    params_hash['method'] = hashmethod
    # Return the hashfunction
    return params_hash


# %%
def url2disk(urls, save_dir):
    """Write url locations to disk.

    Images can also be imported from url locations.
    Each image is first downloaded and stored on a (specified) temp directory.
    In this example we will download 5 images from url locations. Note that url images and path locations can be combined.

    Parameters
    ----------
    urls : list
        list of url locations with image path.
    save_dir : str
        location to disk.

    Returns
    -------
    urls : list of str.
        list to url locations that are now stored on disk.

    Examples
    --------
    >>> # Init with default settings
    >>> import clustimage as cl
    >>>
    >>> # Importing the files files from disk, cleaning and pre-processing
    >>> url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
    >>>                  'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png']
    >>>
    >>> # Import into model
    >>> results = cl.url2disk(url_to_images, r'c:/temp/out/')
    >>>

    """
    return dz.url2disk(urls, save_dir)


# %%
def _set_tempdir(dirpath):
    # Set tempdirectory based on input string or path.
    try:
        # Check directory path
        if dirpath is None:
            dirpath = os.path.join(tempfile.gettempdir(), 'clustimage')
        elif os.path.isdir(dirpath):
            pass
        elif isinstance(dirpath, str):
            dirpath = os.path.join(tempfile.gettempdir(), dirpath)

        # Check existence dir and start clean by removing the directory.
        if not os.path.isdir(dirpath):
            # Create directory
            logger.info('Creating directory: [%s]' %(dirpath))
            os.mkdir(dirpath)
    except:
        raise Exception(logger.error('[%s] does not exists or can not be created.', dirpath))

    dirpath = os.path.abspath(dirpath)
    logger.info("filepath is set to [%s]" %(dirpath))

    return dirpath


# %% Main
# if __name__ == "__main__":
#     import clustimage as clustimage
#     df = clustimage.import_example()
#     out = clustimage.fit(df)
#     fig,ax = clustimage.plot(out)
