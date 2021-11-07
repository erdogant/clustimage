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
from clusteval import clusteval
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

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
# formatter = logging.Formatter('[%(asctime)s] [clustimage]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('[clustimage] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Clustimage():
    """Clustering of images.

    Description
    -----------
    Clustering input images after following steps of pre-processing, feature-extracting, feature-embedding and cluster-evaluation. Taking all these steps requires setting various input parameters.
    Not all input parameters can be changed across the different steps in clustimage. Some parameters are choosen based on best practice, some parameters are optimized, while others are set as a constant.
    The following 4 steps are taken:
        
    Step 1. Pre-processing:
        Images are imported with specific extention (['png','tiff','jpg']) and color-scaled if desired.
        Setting the grayscale parameter to True can be especially usefull when clustering faces.
        Each input images is subsequently scaled in the same dimension such as (128,128). Note that if an array-like dataset is given as an input, setting these dimensions is required to restore the image in case of plotting.
    Step 2. Feature-extraction:
        Features are extracted from the images using Principal component analysis (pca), Histogram of Oriented Gradients (hog) or the raw values are used.
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
        'pca' : PCA feature extraction
        'hog' : hog features extraced
         None : No feature extraction
    embedding : str, (default: 'tsne')
        Perform embedding on the extracted features. The xycoordinates are used for plotting purposes.
        'tsne'
        None
    grayscale : Bool, (default: False)
        Colorscaling the image to gray. This can be usefull when clustering e.g., faces.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    dirpath : str, (default: None)
        Directory to write images.
    ext : list, (default: ['png','tiff','jpg'])
        Images with the file extentions are used.
    params_pca : dict, (default: {'n_components':50, 'detect_outliers':None}.)
        Parameters to initialize the pca model.
    store_to_disk : False, (default: False)
        Required in case of face_detect() or when using an array-like input in fit_transform().
        Images need to be stored on disk to avoid high memory usage.
    verbose : int, (default: 20)
        Print progress to screen. The default is 3.
        60: None, 40: Error, 30: Warn, 20: Info, 10: Debug
    
    Returns
    -------
    object containing results dictionary.
    
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
    >>> cl.scatter(dot_size=50)
    >>> # Plot clustered images
    >>> cl.plot()
    >>>
    >>> # Make prediction
    >>> results_predict = cl.predict(path_to_imgs[0:5], k=None, alpha=0.05)
    >>> cl.plot_predict()
    >>> cl.scatter()
    """

    def __init__(self, method='pca', embedding='tsne', grayscale=False, dim=(128,128), dirpath=None, ext=['png','tiff','jpg'], params_pca={'n_components':50, 'detect_outliers':None}, store_to_disk=False, verbose=20):
        """Initialize clustimage with user-defined parameters."""

        if not np.any(np.isin(method, [None, 'pca','hog'])): raise Exception(logger.error('method: "%s" is unknown', method))
        if dirpath is None: dirpath = tempfile.mkdtemp()
        if not os.path.isdir(dirpath): raise Exception(logger.error('[%s] does not exists.', dirpath))

        # Find path of xml file containing haarcascade file and load in the cascade classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Reads the image as grayscale and results in a 2D-array. In case of RGB, no transparency channel is included.
        self.method = method
        self.embedding = embedding
        self.grayscale = grayscale
        self.cv2_imread_colorscale = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        # self.cv2_imread_colorscale = cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB
        self.dim = dim
        self.dim_face = (64, 64)
        self.params_pca = params_pca
        self.dirpath = dirpath
        self.tempdir = tempfile.mkdtemp()
        self.ext = ext
        self.store_to_disk = store_to_disk # Set if input in np.array with images.
        set_logger(verbose=verbose)

    def fit_transform(self, X, cluster='agglomerative', method='silhouette', metric='euclidean', linkage='ward', min_clust=3, max_clust=25):
        """Import example dataset from github source."""
        # Clean readily fitted models to ensure correct results
        self._clean()
        # Check whether in is dir, list of files or array-like
        X = self.import_data(X)
        # Extract features using method
        raw, X = self.extract_feat(X)
        # Embedding using tSNE
        xycoord = self.compute_embedding(X)
        # Store results
        self.results = {}
        self.results['feat'] = X
        self.results['xycoord'] = xycoord
        self.results['pathnames'] = raw['pathnames']
        self.results['filenames'] = raw['filenames']
        # Cluster
        self.cluster(cluster=cluster, method=method, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, savemem=False)
        # Return
        return self.results

    def import_data(self, Xraw):
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
            # Check dimensions
            pathnames, filenames = None, None
            dim = np.sqrt(len(Xraw[0,:]))
            if (dim!=self.dim[0]) or (dim!=self.dim[1]):
                raise Exception(logger.error('The default dim=%s of the image does not match with the input: %s. Set dim=%s during initialization!' %(str(self.dim), str([int(dim)]*2), str([int(dim)]*2) )))

            if self.store_to_disk:
                # Store images to disk
                pathnames, filenames = [], []
                logger.info('Writing images to tempdir [%s]', self.tempdir)
                for i in tqdm(np.arange(0, Xraw.shape[0])):
                    filename = str(uuid.uuid4())+'.png'
                    pathname = os.path.join(self.tempdir, filename)
                    # Write to disk
                    cv2.imwrite(pathname, Xraw[i,:].reshape(self.dim))
                    filenames.append(filename)
                    pathnames.append(pathname)

            X = {'img': Xraw, 'pathnames':pathnames, 'filenames':filenames}
        return X

    def _clean(self):
        # Clean readily fitted models to ensure correct results.
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if hasattr(self, 'results'): del self.results
            if hasattr(self, 'clusteval'): del self.clusteval
            if hasattr(self, 'results_faces'): del self.results_faces

    def compute_embedding(self, X):
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

    def extract_feat(self, X):
        # If the input is a directory, first collect the images from path
        logger.info('Extracting features using method: [%s]', self.method)
        # Extract features
        if self.method=='pca':
            raw, X = self.extract_pca(X)
        elif self.method=='hog':
            raw, X = self.extract_hog(X)
        else:
            # Read images and preprocessing and flattening of images
            # raw = self.preprocessing(filenames, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=True)
            raw = X.copy()
            X = X['img']

        # Message
        logger.info("Extracted features using [%s]: %s" %(self.method, str(X.shape)))
        return raw, X

    def cluster(self, cluster='agglomerative', method='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, savemem=False, verbose=3):
        if self.results.get('feat', None) is None: raise Exception(logger.error('First run the "fit_transform(pathnames)" function.'))
        logger.info('Cluster evaluation using dataset %s.' %(self.method))
        # Init
        ce = clusteval(cluster=cluster, method=method, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, savemem=False, verbose=3)
        # Fit
        ce.fit(self.results['feat'])
        # Store
        logger.info('Updating cluster labels and evaluated model!')
        self.results['labx'] = ce.results['labx']
        self.clusteval = ce
        # Return
        return ce.results['labx']

    def predict(self, pathnames, metric='euclidean', k=1, alpha=0.05):
        """Import example dataset from github source."""
        if (k is None) and (alpha is None):
            raise Exception(logger.error('Nothing to collect! input parameter "k" and "alpha" can not be None at the same time.'))
        out = None

        # Read images and preprocessing. This is indepdent on the method type but should be in similar manner.
        X = self.preprocessing(pathnames, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=True)

        # Predict according PCA method
        if self.method=='pca':
            Y, dist, feat = self.compute_distances_pca(X, metric=metric, alpha=alpha)
            out = self.collect_pca(X, Y, dist, k, alpha, feat, todf=True)
        else:
            logger.warning('Nothing to predict. Prediction requires initialization with method="pca".')

        # Store
        self.results['predict'] = out
        # Return
        return self.results['predict']

    def preprocessing(self, pathnames, grayscale, dim, flatten=True):
        """Import example dataset from github source."""
        logger.info("Reading images..")
        img, filenames = None, None
        if isinstance(pathnames, str):
            pathnames=[pathnames]
        if isinstance(pathnames, list):
            filenames = list(map(basename, pathnames))
            img = list(map(lambda x: img_read_pipeline(x, grayscale=grayscale, dim=dim, flatten=flatten), tqdm(pathnames)))
            if flatten: img = np.vstack(img)

        out = {}
        out['img'] = img
        out['pathnames'] = pathnames
        out['filenames'] = filenames
        return out

    def extract_hog(self, X):
        # # Read images and preprocessing
        # X = self.preprocessing(pathnames, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False)
        # Set dim correctly for reshaping image
        dim = self.dim if self.grayscale else np.append(self.dim, 3)
        # Extract hog features per image
        feat = list(map(lambda x: hog(x.reshape(dim), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)[1].flatten(), tqdm(X['img'])))
        # Stack all hog features into one array and return
        feat = np.vstack(feat)
        # Return
        return X, feat

    def extract_pca(self, X):
        # Check whether n_components is ok
        if self.params_pca['n_components']>len(X['filenames']): raise Exception(logger.error('n_components should be smaller then the number of samples: %s<%s. Set as following during init: params_pca={"n_components":%s} ' %(self.params_pca['n_components'], len(X['filenames']), len(X['filenames']))))
        # Fit model using PCA
        self.model = pca(**self.params_pca)
        self.model.fit_transform(X['img'], row_labels=X['filenames'])
        # Return
        return X, self.model.results['PC'].values

    # Compute distances and probabilities after transforming the data using PCA.
    def compute_distances_pca(self, X, metric, alpha):
        dist = None
        # Transform new "unseen" data. Note that these datapoints are not really unseen as they are readily fitted above.
        PCnew = self.model.transform(X['img'], row_labels=X['filenames'])
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
                # If alpha is not used, set all to nan
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

    def detect_faces(self, pathnames):
        # If face detection, grayscale should be True.
        if (not self.grayscale): logger.warning('It is advisable to set "grayscale=True" when detecting faces.')

        # Read and pre-proces the input images
        logger.info("Read images>")
        # Create empty list
        faces = {'img':[], 'pathnames':[], 'filenames':[], 'facepath':[], 'coord_faces':[], 'coord_eyes':[]}
        # Extract faces and eyes from image
        for pathname in pathnames:
            # Extract faces
            facepath, imgfaces, coord_faces, coord_eyes, filename, path_to_image = self.extract_faces(pathname)
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
    
    def extract_faces(self, pathname):
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
    
    def plot_faces(self, faces=True, eyes=True, cmap=None):
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
        if hasattr(self, 'clusteval'):
            self.clusteval.plot()
            results = self.clusteval.dendrogram(max_d=max_d, figsize=figsize)
            # if feat is None: feat=self.results['feat']
            # results = self.clusteval.dendrogram(X=feat, max_d=max_d, figsize=figsize)
        return results

    def scatter(self, dot_size=15, legend=False, figsize=(15,10)):
        # Set default settings
        labx = self.results.get('labx', None)
        if labx is None: labx=np.zeros_like(self.results['xycoord'][:,0]).astype(int)

        # Scatter cluster evaluation
        if hasattr(self, 'clusteval'):
            self.clusteval.scatter(self.results['feat'])

        if self.embedding=='tsne':
            from scatterd import scatterd
            colours=np.vstack(colourmap.fromlist(labx)[0])
            fig, ax = scatterd(self.results['xycoord'][:,0], self.results['xycoord'][:,1], s=dot_size, c=colours, label=labx, figsize=figsize, title='tSNE plot')

        # Scatter all points
        if self.method=='pca':
            fig, ax = self.model.plot(figsize=figsize)
            fig, ax = self.model.scatter(y=labx, legend=legend, label=False, figsize=figsize)
            # fig, ax = self.model.scatter3d(y=1, legend=legend, label=False, figsize=figsize)

        # Scatter the predicted cases
        if self.results.get('predict', None) is not None:
            fig, ax = self.model.scatter(y=labx, legend=legend, label=False, figsize=figsize)
            # Create unique colors
            colours = colourmap.fromlist(self.results['predict']['feat'].index)[1]
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    x,y,z = self.results['predict']['feat'].iloc[:,0:3].loc[key]
                    idx = self.results['predict'][key]['y_idx']
                    # Scatter
                    ax.scatter(x, y, color=colours[key], edgecolors=[0,0,0])
                    ax.text(x,y, key, color=colours[key])
                    ax.scatter(self.results['feat'][idx][:,0], self.results['feat'][idx][:,1], edgecolors=[0,0,0])

    def plot_predict(self, cmap=None, figsize=(15,10)):
        cmap = _set_cmap(cmap, self.grayscale)
        # Plot the images that are similar to each other.
        if self.results.get('predict', None) is not None:
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    # Collect images
                    input_img = self.results['predict'][key]['x_path']
                    predict_img = self.results['predict'][key]['y_path']
                    # Input label
                    if isinstance(input_img, str): input_img=[input_img]
                    # Input images
                    I_input = list(map(lambda x: img_read_pipeline(x, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False), input_img))
                    # Predicted label
                    I_predict = list(map(lambda x: img_read_pipeline(x, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False), predict_img))
                    # Make the real plot
                    title='Top or top-left image is input. The others are predicted.'

                    imgs=I_input+I_predict
                    self._make_subplots(imgs, None, cmap, figsize, title)

                    # fig, axes = plt.subplots(len(I_predict)+1,1,sharex=True,sharey=True,figsize=(8,10))
                    # axes[0].set_title('Input image')
                    # axes[0].imshow(I_input[0], cmap=cmap)
                    # axes[0].axis('off')
                    # for i, I in enumerate(I_predict):
                    #     axes[i+1].set_title('Predicted: %s' %(i+1))
                    #     axes[i+1].imshow(I, cmap=cmap)
                    #     axes[i+1].axis('off')

    def _make_subplots(self, imgs, ncols, cmap, figsize, title=''):
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

    def plot(self, ncols=10, legend=False, cmap=None, show_hog=False, figsize=(15,10)):
        # Set cmap
        cmap = _set_cmap(cmap, self.grayscale)
        # Plot the clustered images
        if (self.results.get('labx', None) is not None) and (self.results.get('pathnames', None) is not None):
            uilabx = np.unique(self.results['labx'])
            for labx in tqdm(uilabx):
                idx = np.where(self.results['labx']==labx)[0]
                # Collect the images
                getfiles = np.array(self.results['pathnames'])[idx]
                # Get the images that cluster together
                imgs = list(map(lambda x: img_read_pipeline(x, grayscale=self.cv2_imread_colorscale, dim=self.dim, flatten=False), getfiles))
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
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # assert img.shape[0:2]==dim
    return img


# %% Convert into grayscale
# def to_gray(img):
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     except:
#         gray = img
#     return gray

# %% Set cmap
def _set_cmap(cmap, grayscale):
    if cmap is None:  cmap = 'gray' if grayscale else None
    cmap = 'gray' if grayscale else None
    return cmap

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
def import_example(data='flowers', url=None):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'flowers', 'faces'
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
    logger.info('[%s] files are collected recursively from path: [%s]' %(len(getfiles), dirpath))
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
