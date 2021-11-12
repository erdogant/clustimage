from clustimage import Clustimage
import itertools as it
import numpy as np
import unittest

class TestCLUSTIMAGE(unittest.TestCase):

    def test_fit_transform(self):
        # Example data
        cl = Clustimage()
        Xflowers = cl.import_example(data='flowers')
        Xdigits = cl.import_example(data='digits')
        Xfaces = cl.import_example(data='faces')
        # Parameters combinations to check
        param_grid = {
        	'method':['pca', 'hog', None],
        	'embedding':['tsne', None],
        	'cluster_space' : ['high', 'low'],
        	'grayscale' : [True, False],
            'dim' : [(8,8), (128,128), (256,256)],
            'data' : [Xflowers, Xdigits, Xfaces]
        	}
        # Make the combinatinos
        allNames = param_grid.keys()
        combinations = list(it.product(*(param_grid[Name] for Name in allNames)))
        # Iterate over all combinations
        for combination in combinations:
            # init
            cl = Clustimage(method=combination[0], embedding=combination[1], cluster_space=combination[2], grayscale=combination[3], dim=combination[4], params_pca={'n_components':50, 'detect_outliers':None})
            # Preprocessing and feature extraction
            assert cl.fit_transform(combination[5])
    
    def test_cluster(self):
        cl = Clustimage()
        X = cl.import_example(data='flowers')
        results = cl.fit_transform(X)
        assert np.all(np.isin([*cl.results.keys()], ['feat', 'xycoord', 'pathnames', 'filenames', 'labx']))
        assert len(cl.cluster())==len(X)
    
        # Parameters combinations to check
        param_grid = {
        	'cluster_space':['high','low'],
        	'cluster':['agglomerative'],
        	'method' : ['silhouette', 'dbindex'],
            'min_clust' : [2, 4, 6],
            'max_clust' : [10, 20, 30],
        	}
        # Make the combinatinos
        allNames = param_grid.keys()
        combinations = list(it.product(*(param_grid[Name] for Name in allNames)))
        for combination in combinations:
            labx = cl.cluster(cluster_space=combination[0], cluster=combination[1], method=combination[2], metric='euclidean', linkage='ward', min_clust=combination[3], max_clust=combination[4])
            assert len(labx)==len(X)
    
    def test_find(self):
        cl = Clustimage(method='pca', embedding='tsne', grayscale=False)
        # load example with flowers
        path_to_imgs = cl.import_example(data='flowers')
        # Extract features (raw images are not stored and handled per-image to save memory)
        results = cl.fit_transform(path_to_imgs, min_clust=10)
    
        # Predict
        results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
        assert np.all(np.isin([*results_find.keys()], ['feat', '0001.png', '0002.png', '0003.png', '0004.png', '0005.png']))
        assert len(results_find['0001.png']['y_idx'])==2
        assert len(results_find['0002.png']['y_idx'])==1
        assert len(results_find['0003.png']['y_idx'])==30
        assert len(results_find['0004.png']['y_idx'])==2
        assert len(results_find['0005.png']['y_idx'])==1
        
        results_find = cl.find(path_to_imgs[0:5], k=1, alpha=None)
        assert len(results_find['0001.png']['y_idx'])==1
        assert len(results_find['0002.png']['y_idx'])==1
        assert len(results_find['0003.png']['y_idx'])==1
        assert len(results_find['0004.png']['y_idx'])==1
        assert len(results_find['0005.png']['y_idx'])==1
    
    def test_predict(self):
        # Init
        cl = Clustimage(method='pca', grayscale=True, params_pca={'n_components':14})
        # Load example with faces
        pathnames = cl.import_example(data='faces')
        # Detect faces
        face_results = cl.detect_faces(pathnames)
        assert np.all(np.isin([*cl.results_faces.keys()], ['img', 'pathnames', 'filenames', 'facepath', 'coord_faces', 'coord_eyes']))
        # Cluster
        results = cl.fit_transform(face_results['facepath'])
        assert np.all(np.isin([*cl.results.keys()], ['feat', 'xycoord', 'pathnames', 'filenames', 'labx']))
