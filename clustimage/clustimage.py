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
import colourmap
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
import logging
from urllib.parse import urlparse
import fnmatch
import zipfile
import requests
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

# Configure the logger
logger = logging.getLogger('')
for handler in logger.handlers[:]: # get rid of existing old handlers
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
            * 'pca-hog' : PCA extracted features from the HOG desriptor
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
    >>> X = cl.import_example(data='mnist')
    >>>
    >>> # Cluster digits
    >>> results = cl.fit_transform(X)
    >>>
    >>> # Cluster evaluation
    >>> cl.clusteval.plot()
    >>> cl.clusteval.scatter(cl.results['xycoord'])
    >>> cl.pca.plot()
    >>>
    >>> # Unique
    >>> cl.plot_unique(img_mean=False)
    >>> cl.results_unique.keys()
    >>>
    >>> # Scatter
    >>> cl.scatter(img_mean=False, zoom=3)
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
    def __init__(self, method='pca', embedding='tsne', grayscale=False, dim=(128,128), dim_face=(64,64), dirpath=None, store_to_disk=True, ext=['png','tiff','jpg'], params_pca={'n_components':0.95}, params_hog={'orientations':8, 'pixels_per_cell':(8,8), 'cells_per_block':(1,1)}, verbose=20):
        """Initialize clustimage with user-defined parameters."""
        # Clean readily fitted models to ensure correct results
        self._clean()

        if not np.any(np.isin(method, [None, 'pca','hog', 'pca-hog'])): raise Exception(logger.error('method: "%s" is unknown', method))
        if dirpath is None: dirpath = tempfile.mkdtemp()
        if not os.path.isdir(dirpath): raise Exception(logger.error('[%s] does not exists.', dirpath))

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

        self.params['dirpath'] = dirpath
        self.params['tempdir'] = tempfile.mkdtemp()
        self.params['ext'] = ext
        self.params['store_to_disk'] = store_to_disk

        pca_defaults = {'n_components':0.95, 'detect_outliers': None, 'random_state': None}
        params_pca   = {**pca_defaults, **params_pca}
        self.params_pca = params_pca

        hog_defaults = {'orientations':8, 'pixels_per_cell':(8,8), 'cells_per_block':(1,1)}
        params_hog   = {**hog_defaults, **params_hog}
        self.params_hog = params_hog

        set_logger(verbose=verbose)

    def fit_transform(self, X, cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=3, max_clust=25, cluster_space='high'):
        """Group samples into clusters that are similar in their feature space.
        
        Description
        -----------
        The fit_transform function allows to detect natural groups or clusters of images. It works using a multi-step proces of pre-processing, extracting the features, and evaluating the optimal number of clusters across the feature space.
        The optimal number of clusters are determined using well known methods suchs as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
        Based on the clustering results, the unique images are also gathered.

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
        >>> pathnames = cl.import_example(data='faces')
        >>> # Detect faces
        >>> face_results = cl.detect_faces(pathnames)
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
        self.cluster(cluster=cluster, evaluate=evaluate, cluster_space=cluster_space, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust)
        # Return
        return self.results

    def cluster(self, cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=3, max_clust=25, cluster_space='high'):
        """Detection of the optimal number of clusters given the input set of features.
        
        Description
        -----------
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
        >>> # If you want to cluster on the low-dimensional space 
        >>> labels = cl.cluster(min_clust=5, max_clust=25, cluster_space='low', cluster='dbscan')
        >>> cl.scatter(dotsize=50, img_mean=False)
        >>>

        """
        if self.results.get('feat', None) is None: raise Exception(logger.error('First run the "fit_transform(pathnames)" function.'))
        self.params['cluster_space'] = cluster_space

        # Init
        ce = clusteval(cluster=cluster, evaluate=evaluate, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, verbose=3)

        # Fit
        if cluster_space=='low':
            feat = self.results['xycoord']
            logger.info('Cluster evaluation using the [%s] feature space of the [%s] coordinates.', cluster_space, self.params['embedding'])
        else:
            feat = self.results['feat']
            logger.info('Cluster evaluation using the [%s] feature space of the [%s] features.', cluster_space, self.params['method'])

        # Fit model
        _ = ce.fit(feat)

        # Store results and params
        logger.info('Updating cluster-labels and cluster-model based on the %s feature-space.', str(feat.shape))
        self.results['labels'] = ce.results['labx']
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
            raise Exception(logger.error('Results in missing! Hint: try to first fit_transform() your data!'))

    def unique(self, metric=None):
        """Compute the unique images.

        Description
        -----------
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
        >>> X = cl.import_example(data='mnist')
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
        if metric is None: metric=self.params_clusteval['metric']
        eigen_img, pathnames, center_idx, center_coord = [], [], [], []
        # Unique labels
        uilabels = np.unique(self.results['labels'])

        # Run over all cluster labels
        for label in uilabels:
            # Get cluster label
            idx = np.where(self.results['labels']==label)[0]
            # Compute center of cluster
            # self.results['feat'][idx,:].mean(axis=0)
            xycoord_center = np.mean(self.results['xycoord'][idx,:], axis=0)

            # Compute the average image by simply averaging the images
            img = np.vstack(list(map(lambda x: self.imread(x, colorscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=True), np.array(self.results['pathnames'])[idx])))
            eigen_img.append(imscale(np.mean(img, axis=0)))

            # dim = _check_dim(eigen_img, self.params['dim'])
            # plt.figure();plt.imshow(eigen_img.reshape(dim))

            # Compute distance across all samples
            dist = distance.cdist(self.results['xycoord'], xycoord_center.reshape(-1,1).T, metric=metric)
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
        self.results_unique = {'labels':uilabels, 'idx':center_idx, 'xycoord_center':np.vstack(center_coord), 'pathnames':pathnames, 'img_mean':np.vstack(eigen_img)}
        return self.results_unique
        
    def find(self, Xnew, metric=None, k=None, alpha=0.05):
        """Find images that are similar to that of the input image.

        Description
        -----------
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
        >>> X = cl.import_example(data='mnist')
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
        if (k is None) and (alpha is None):
            raise Exception(logger.error('Nothing to collect! input parameter "k" and "alpha" can not be None at the same time.'))
        if metric is None: metric=self.params_clusteval['metric']
        logger.info('Find similar images with metric [%s], k-nearest neighbors: %s and under alpha: %s ' %(metric, str(k), str(alpha)))

        # Check whether in is dir, list of files or array-like
        Xnew = self._import_data(Xnew)
        # Compute distance
        Y, feat = self._compute_distances(Xnew, metric=metric, alpha=alpha)
        # Collect the samples
        out = self._collect_pca(Xnew, Y, k, alpha, feat, todf=True)

        # Store
        self.results['predict'] = out
        self.params['metric_find'] = metric
        return self.results['predict']

    def detect_faces(self, pathnames):
        """Detect and extract faces from images.
        
        Description
        -----------
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
        >>> # load example with faces
        >>> pathnames = cl.import_example(data='faces')
        >>>
        >>> # Detect faces
        >>> face_results = cl.detect_faces(pathnames)
        >>>
        >>> # Cluster the faces
        >>> results = cl.fit_transform(face_results['pathnames_face'])
        >>>
        >>> # Plot facces
        >>> cl.plot_faces(faces=True, eyes=True)
        >>>

        """
        # If face detection, grayscale should be True.
        if (not self.params['grayscale']): logger.warning('It is advisable to set "grayscale=True" when detecting faces.')

        # Read and pre-proces the input images
        X = self._import_data(pathnames, imread=False)
        # Create empty list
        faces = {'img':[], 'pathnames':[], 'filenames':[], 'pathnames_face':[], 'coord_faces':[], 'coord_eyes':[]}
        
        # Set logger to warning-error only
        verbose = logger.getEffectiveLevel()
        set_logger(verbose=30)

        # Extract faces and eyes from image
        for pathname in tqdm(X['pathnames'], disable=disable_tqdm()):
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

    def preprocessing(self, pathnames, grayscale, dim, imread=True, flatten=True):
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
            if imread:
                img = list(map(lambda x: self.imread(x, colorscale=grayscale, dim=dim, flatten=flatten), tqdm(pathnames, disable=disable_tqdm())))
                if flatten: img = np.vstack(img)

        out = {}
        out['img'] = img
        out['pathnames'] = pathnames
        out['filenames'] = filenames
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
        # If 1D-vector, make 2D-array
        if len(X.shape)==1: X = X.reshape(-1,1).T
        # Set dim correctly for reshaping image
        dim = _check_dim(X, self.params['dim'], grayscale=self.params['grayscale'])
        # Extract hog features per image
        if flatten:
            feat = list(map(lambda x: hog(x.reshape(dim), orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)[1].flatten(), tqdm(X, disable=disable_tqdm())))
            # Stack all hog features in NxM array
            feat = np.vstack(feat)
        else:
            feat = list(map(lambda x: hog(x.reshape(dim), orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)[1], tqdm(X, disable=disable_tqdm())))[0]
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

    def _import_data(self, Xraw, imread=True, flatten=True):
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
            Xraw = listdir(Xraw, ext=self.params['ext'])
            logger.info('Extracted images: [%s]', len(Xraw))
        # Check string
        if isinstance(Xraw, str) and os.path.isfile(Xraw):
            Xraw = [Xraw]
        # Check numpy array
        # if type(Xraw).__module__ == np.__name__:
        #     Xraw = list(Xraw)

        # 2. Read images
        if isinstance(Xraw, list):
            # Make sure that list in lists are flattend
            Xraw = list(np.hstack(Xraw))
            # Read images and preprocessing
            X = self.preprocessing(Xraw, grayscale=self.params['cv2_imread_colorscale'], dim=self.params['dim'], flatten=flatten, imread=imread)

        # 3. If input is array-like. Make sure X becomes compatible.
        if isinstance(Xraw, np.ndarray):
            # Make 2D
            if len(Xraw.shape)==1:
                Xraw = Xraw.reshape(-1,1).T
            # Check dimensions
            pathnames, filenames = None, None
            # Check dim
            self.params['dim'] = _check_dim(Xraw, self.params['dim'])
            # Store to disk
            if self.params['store_to_disk']:
                pathnames, filenames = store_to_disk(Xraw, self.params['dim'], self.params['tempdir'])

            # Make dict
            X = {'img': Xraw, 'pathnames':pathnames, 'filenames':filenames}
        return X

    def _clean(self):
        """Clean or removing previous results and models to ensure correct working."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if hasattr(self, 'results'): del self.results
            if hasattr(self, 'results_faces'): del self.results_faces
            if hasattr(self, 'results_unique'): del self.results_unique
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
        if self.params['embedding']=='tsne':
            logger.info('Computing embedding using %s..', self.params['embedding'])
            xycoord = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)
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
        logger.info('Extracting features using method: [%s]', self.params['method'])
        # Extract features
        if self.params['method']=='pca':
            X = self.extract_pca(Xraw)
        elif self.params['method']=='hog':
            X = self.extract_hog(Xraw['img'], orientations=self.params_hog['orientations'], pixels_per_cell=self.params_hog['pixels_per_cell'], cells_per_block=self.params_hog['cells_per_block'])
        elif self.params['method']=='pca-hog':
            X = {}
            X['img'] = self.extract_hog(Xraw['img'], orientations=self.params_hog['orientations'], pixels_per_cell=self.params_hog['pixels_per_cell'], cells_per_block=self.params_hog['cells_per_block'])
            X['filenames'] = Xraw['filenames']
            X = self.extract_pca(X)
        else:
            # Read images and preprocessing and flattening of images
            X = Xraw['img'].copy()

        # Message
        logger.info("Extracted features using [%s]: %s" %(self.params['method'], str(X.shape)))
        return Xraw, X

    def _compute_distances(self, X, metric, alpha):
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
            _, Xmapped = self._extract_feat(X)
            # Compute distance from input sample to all other samples
        Y = distance.cdist(self.results['feat'], Xmapped, metric=metric)

        # Sanity check
        if np.any(np.isnan(Y)):
            logger.warning('The metric [%s] results in NaN! Please change metric for appropriate results!', metric)

        # Fit distribution to emperical data and compute probability of the distances of interest
        if (alpha is not None) and ( (not hasattr(self,'distfit')) or (self.params['metric_find'] != metric) ):
            # Compute distance across all samples
            Ytot = distance.cdist(self.results['feat'], self.results['feat'], metric=metric)
            # Take a subset of samples to prevent high computation times.
            x_max, y_max = np.minimum(500, Ytot.shape[0]), np.minimum(500, Ytot.shape[1])
            xrow, yrow = random.sample(range(1, x_max), x_max-1), random.sample(range(1, y_max), y_max-1)
            Ytot = Ytot[xrow, :]
            Ytot = Ytot[:, yrow]
            # Init distfit
            self.distfit = distfit(bound='down', multtest=None, distr=['norm', 'expon', 'uniform', 'gamma', 't'])
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
        for (x,y,w,h) in coord_faces:
            # Create filename for face
            filename = os.path.join(self.params['dirpath'], str(uuid.uuid4()))+'.png'
            pathnames_face.append(filename)
            # Store faces seperately
            imgface = imresize(img[y:y+h, x:x+w], dim=self.params['dim_face'])
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
                idx_k = np.argsort(Y[:,i])[0:k]
            # Collect samples based on probability
            if alpha is not None:
                dist_results = self.distfit.predict(Y[:,i], verbose=0)
                idx_dist = np.where(dist_results['y_proba']<=alpha)[0]
                # Sort on significance
                idx_dist = idx_dist[np.argsort(dist_results['y_proba'][idx_dist])]
            else:
                # If alpha is not used, set all to nan
                dist_results={}
                dist_results['y_proba'] = np.array([np.nan]*Y.shape[0])

            # Combine the unique k-nearest samples and probabilities.
            idx = unique_no_sort(np.append(idx_dist, idx_k))
            # Store in dict
            logger.info('[%d] similar images found for [%s]' %(len(idx), filename))
            store_key = {**store_key, 'y_idx': idx, 'distance': Y[idx, i], 'y_proba': dist_results['y_proba'][idx], 'y_filenames': np.array(self.results['filenames'])[idx].tolist(), 'y_pathnames': np.array(self.results['pathnames'])[idx].tolist(), 'x_pathnames': X['pathnames'][i]}
            if todf: store_key = pd.DataFrame(store_key)
            out[filename] = store_key

        # Return
        return out

    def imread(self, filepath, colorscale=1, dim=(128, 128), flatten=True):
        """Read and pre-processing of images.

        Description
        -----------
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
        ---------
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
        # Read the image
        img = _imread(filepath, colorscale=colorscale)
        # Scale the image
        img = imscale(img)
        # Resize the image
        img = imresize(img, dim=dim)
        # Flatten the image
        if flatten: img = img_flatten(img)
        # Return
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
        if (filepath is None) or (filepath==''):
            filepath = 'clustimage.pkl'
        if filepath[-4:] != '.pkl':
            filepath = filepath + '.pkl'
        # Store data
        storedata = {}
        storedata['results'] = self.results
        storedata['params'] = self.params
        storedata['params_pca'] = self.params_pca
        storedata['params_hog'] = self.params_hog
        if hasattr(self,'results_faces'): storedata['results_faces'] = self.results_faces
        if hasattr(self,'results_unique'): storedata['results_unique'] = self.results_unique
        if hasattr(self,'distfit'): storedata['distfit'] = self.distfit
        if hasattr(self,'clusteval'): storedata['clusteval'] = self.clusteval
        # if hasattr(self,'pca'): storedata['pca'] = self.pca
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
        if (filepath is None) or (filepath==''):
            filepath = 'clustimage.pkl'
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'

        # Load
        storedata = pypickle.load(filepath, verbose=verbose)

        # Store in self
        if storedata is not None:
            self.results = storedata['results']
            self.params = storedata['params']
            self.params_pca = storedata['params_pca']
            self.params_hog = storedata['params_hog']
            self.results_faces = storedata.get('results_faces', None)
            self.results_unique = storedata.get('results_unique', None)
            self.distfit = storedata.get('distfit', None)
            self.clusteval = storedata.get('clusteval', None)
            self.pca = storedata.get('pca', None)

            logger.info('Load succesful!')
            # Return results
            # return self.results
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
        cmap = _set_cmap(cmap, self.params['grayscale'])
        # Set logger to warning-error only
        verbose = logger.getEffectiveLevel()
        set_logger(verbose=30)

        # Walk over all detected faces
        if hasattr(self, 'results_faces'):
            for i, pathname in tqdm(enumerate(self.results_faces['pathnames']), disable=disable_tqdm()):
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
                    for k in np.arange(0, len(self.results_faces['pathnames_face'][i])):
                        # face = self.results_faces['img'][i][k].copy()
                        pathnames_face = self.results_faces['pathnames_face'][i][k]
                        if os.path.isfile(pathnames_face):
                            face = self.preprocessing(pathnames_face, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()
                            if coord_eyes[k] is not None:
                                plt.figure()
                                for (ex,ey,ew,eh) in coord_eyes[k]:
                                    cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                                if len(face.shape)==3:
                                    plt.imshow(face[:,:,::-1]) # RGB-> BGR
                                else:
                                    plt.imshow(face)
                        else:
                            logger.warning('File is removed: %s', pathnames_face)

                # Pause to plot to screen
                plt.pause(0.1)
        else:
            logger.warning('Nothing to plot. First detect faces with ".detect_faces(pathnames)"')
        
        set_logger(verbose=verbose)

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
        self._check_status()

        if hasattr(self, 'clusteval'):
            results = self.clusteval.dendrogram(max_d=max_d, figsize=figsize)
            # results = self.clusteval.dendrogram(X=feat, max_d=max_d, figsize=figsize)
        else:
            logger.warning('This Plot requires running fit_transform() first.')

        if max_d is not None:
            return results

    def _add_img_to_scatter(self, ax, pathnames, xycoord, cmap=None, zoom=0.2):
        # Plot the images on top of the scatterplot
        if zoom is not None:
            for i, pathname in enumerate(pathnames):
                if isinstance(pathname, str):
                    img = self.imread(pathname, dim=self.params['dim'], colorscale=self.params['cv2_imread_colorscale'], flatten=False)
                else:
                    dim = _check_dim(pathname, self.params['dim'], grayscale=self.params['grayscale'])
                    # dim = _check_dim(pathname, self.params['dim'])
                    # plt.figure();plt.imshow(eigen_img.reshape(dim))
                    img = pathname.reshape(dim)
                # Make hte plot
                imagebox = offsetbox.AnnotationBbox( offsetbox.OffsetImage(img, cmap=cmap, zoom=zoom), xycoord[i,:] )
                ax.add_artist(imagebox)

    def scatter(self, dotsize=15, legend=False, zoom=0.3, img_mean=True, figsize=(15,10)):
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
        # Check status
        self._check_status()
        # Set default settings
        cmap = plt.cm.gray if self.params['grayscale'] else None
        # Set logger to warning-error only
        verbose = logger.getEffectiveLevel()
        set_logger(verbose=40)
        # Get the cluster labels
        labels = self.results.get('labels', None)
        if labels is None: labels=np.zeros_like(self.results['xycoord'][:,0]).astype(int)

        # Make scatterplot
        colours=np.vstack(colourmap.fromlist(labels)[0])
        title = ('tSNE plot. Samples are coloured on the cluster labels (%s dimensional).' %(self.params['cluster_space']))
        fig, ax = scatterd(self.results['xycoord'][:,0], self.results['xycoord'][:,1], s=dotsize, c=colours, label=labels, figsize=figsize, title=title, fontsize=18, fontcolor=[0,0,0], xlabel='x-axis', ylabel='y-axis')

        if hasattr(self, 'results_unique'):
            if img_mean:
                X = self.results_unique['img_mean']
            else:
                X = self.results_unique['pathnames']
            self._add_img_to_scatter(ax, cmap=cmap, zoom=zoom, pathnames=X, xycoord=self.results_unique['xycoord_center'])

        # Scatter the predicted cases
        if (self.results.get('predict', None) is not None):
            if self.params['method']=='pca':
                # Scatter all points
                fig, ax = self.pca.scatter(y=labels, legend=legend, label=False, figsize=figsize)
                # Create unique colors
                colours = colourmap.fromlist(self.results['predict']['feat'].index)[1]
                for key in self.results['predict'].keys():
                    if self.results['predict'].get(key).get('y_idx', None) is not None:
                        x = self.results['predict']['feat'].iloc[:,0].loc[key]
                        if len(self.results['predict']['feat'])>=2:
                            y = self.results['predict']['feat'].iloc[:,1].loc[key]
                        else:
                            y=0
                        idx = self.results['predict'][key]['y_idx']
                        # Scatter
                        ax.scatter(x, y, color=colours[key], edgecolors=[0,0,0])
                        ax.text(x,y, key, color=colours[key])
                        if self.results['feat'].shape[1]>=2:
                            ax.scatter(self.results['feat'][idx][:,0], self.results['feat'][idx][:,1], edgecolors=[0,0,0])
                                
            else:
                logger.info('Mapping predicted results is only possible when uing method="pca".')

        # Restore verbose status
        set_logger(verbose=verbose)

    def plot_unique(self, cmap=None, img_mean=True, show_hog=False, figsize=(15,10)):
        if hasattr(self, 'results_unique') is not None:
            # Set logger to warning-error only
            verbose = logger.getEffectiveLevel()
            set_logger(verbose=40)
            # Defaults
            imgs, imgshog = [], []
            cmap = _set_cmap(cmap, self.params['grayscale'])
            txtlabels = list(map(lambda x: 'Cluster'+x, self.results_unique['labels'].astype(str)))

            # Collect the image data
            if img_mean:
                subtitle='(averaged per cluster)'
                imgs=self.results_unique['img_mean']
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
                        hogtmp = exposure.rescale_intensity(self.results['feat'][idx,:].reshape(self.params['dim']), in_range=(0,10))
                        imgshog.append(hogtmp)

            self._make_subplots(imgs, None, cmap, figsize, title='Unique images '+subtitle, labels=txtlabels)

            if show_hog and (self.params['method']=='hog'):
                self._make_subplots(imgshog, None, 'binary', figsize, title='Unique HOG images '+subtitle, labels=txtlabels)

            # Restore verbose status
            set_logger(verbose=verbose)
        else:
            logger.warning('Plotting unique images is not possible. Hint: Try to run the unique() function first.')

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
                            labels = ['Input'] + list(map(lambda x: 'k='+x, np.arange(1,len(I_find)+1).astype(str)))
                        title = 'Find similar images for [%s].' %(input_txt)
                        # Make the subplot
                        self._make_subplots(imgs, None, cmap, figsize, title=title, labels=labels)
                        logger.info('[%d] similar images detected for input image: [%s]' %(len(find_img), key))
                except:
                    pass
        else:
            logger.warning('No prediction results are found. Hint: Try to run the .find() functionality first.')

    def plot(self, labels=None, show_hog=False, ncols=None, cmap=None, figsize=(15,10)):
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
            # Gather labels
            if labels is None: labels = self.results['labels']
            if not isinstance(labels, list): labels = [labels]
            # Unique labels
            uilabels = np.unique(labels)

            # Run over all labels.
            for label in tqdm(uilabels, disable=disable_tqdm()):
                idx = np.where(self.results['labels']==label)[0]
                if len(idx)>0:
                    # Collect the images
                    getfiles = np.array(self.results['pathnames'])[idx]
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
                        hog_images = self.results['feat'][idx,:]
                        fig, axs = plt.subplots(len(imgs), 2, figsize=(15,10), sharex=True, sharey=True)
                        ax = axs.ravel()
                        fignum=0
                        for i, img in enumerate(imgs):
                            hog_image_rescaled = exposure.rescale_intensity(hog_images[i,:].reshape(self.params['dim']), in_range=(0,10))
                            ax[fignum].imshow(img, cmap=cmap)
                            ax[fignum].axis('off')
                            ax[fignum+1].imshow(hog_image_rescaled, cmap=cmap)
                            ax[fignum+1].axis('off')
                            fignum=fignum+2
    
                        _ = fig.suptitle('Histogram of Oriented Gradients', fontsize=16)
                        plt.tight_layout()
                        plt.show()
                else:
                    logger.error('The cluster clabel [%s] does not exsist! Skipping!', label)
        else:
            logger.warning('Plotting is not possible. Path locations are unknown. Hint: try to set "store_to_disk=True" during initialization.')
    
    def _get_rows_cols(self, n, ncols=None):
        # Setup rows and columns
        if ncols is None: ncols=np.maximum(int(np.ceil(np.sqrt(n))), 2)
        nrows = int(np.ceil(n/ncols))
        return nrows, ncols

    def _make_subplots(self, imgs, ncols, cmap, figsize, title='', labels=None):
        """Make subplots."""
        # Get appropriate dimension
        dim = self.params['dim'] if self.params['grayscale'] else np.append(self.params['dim'], 3)
        # Setup rows and columns
        nrows, ncols = self._get_rows_cols(len(imgs), ncols=ncols)
        # Make figure
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
        # Make the actual plots
        for i, ax in enumerate(axs.ravel()):
            if i<len(imgs):
                if len(imgs[i].shape)==1:
                    ax.imshow(imgs[i].reshape(dim), cmap=cmap)
                else:
                    ax.imshow(imgs[i], cmap=cmap)
                if labels is not None: ax.set_title(labels[i])
            ax.axis("off")
        _ = fig.suptitle(title, fontsize=16)

        # Small pause to build the plot
        plt.pause(0.1)
        # Return the rows and columns
        return nrows, ncols

    def clean_files(self):
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
        logger.debug('The dim is changed into: %s', str(dim))

    return dim

# %% Store images to disk
def store_to_disk(Xraw, dim, tempdir):
    """Store to disk."""
    # Determine the dimension based on the length of the vector.
    dim = _check_dim(Xraw, dim)
    # Store images to disk
    pathnames, filenames = [], []
    logger.info('Writing images to tempdir [%s]', tempdir)
    for i in tqdm(np.arange(0, Xraw.shape[0]), disable=disable_tqdm()):
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
def imresize(img, dim=(128, 128)):
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
def imscale(img):
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
    if os.path.isfile(filepath):
        # Read the image
        img = cv2.imread(filepath, colorscale)
    else:
        logger.warning('File does not exists: %s', filepath)
    
    # In case of rgb images: make gray images compatible with RGB
    if ((colorscale!=0) and (colorscale!=6)) and (len(img.shape)<3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

# %%
def set_logger(verbose=20):
    """Set the logger for verbosity messages."""
    logger.setLevel(verbose)


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


# %% Import example dataset from github.
def import_example(data='flowers', url=None):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'flowers', 'faces', 'mnist'
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
        elif data=='mnist':
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
    dirpath = unzip(path_to_data)
    # Import local dataset
    image_files = listdir(dirpath)
    # Return
    return image_files


# %% Recursively list files from directory
def listdir(dirpath, ext=['png','tiff','jpg']):
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

    Example
    -------
    >>> import clustimage as cl
    >>> pathnames = cl.listdir('c://temp//flower_images')

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
def unzip(path_to_zip):
    """Unzip files.

    Parameters
    ----------
    path_to_zip : str
        Path of the zip file.

    Returns
    -------
    getpath : str
        Path containing the unzipped files.

    Example
    -------
    >>> import clustimage as cl
    >>> dirpath = cl.unzip('c://temp//flower_images.zip')

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

    Example
    -------
    >>> import clustimage as cl
    >>> images = cl.wget('https://erdogant.github.io/datasets/flower_images.zip', 'c://temp//flower_images.zip')

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
