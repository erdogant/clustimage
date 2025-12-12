from clustimage import Clustimage
import itertools as it
import numpy as np
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

class TestCLUSTIMAGE(unittest.TestCase):

    def test_part1(self):
        # Import libraries
        from clustimage import Clustimage
        import matplotlib.pyplot as plt

        # Initialize
        cl = Clustimage()
        
        # Load example dataset
        pathnames = cl.import_example(data='flowers')
        
        # Preprocessing of the first image
        # 0: cv2.IMREAD_GRAYSCALE
        # 1: cv2.IMREAD_COLOR
        img = cl.imread(pathnames[0], dim=(128,128), colorscale=1, flatten=True)
        
        # Flattened array
        assert img.shape == (49152,)
        
        # Plot. Note that reshape is only required in case flatten=True
        plt.figure()
        plt.imshow(img.reshape(128,128,3))
        plt.axis('off')
    
    def  test_part2(self):
        # Initialize
        cl = Clustimage(method='hog', grayscale=False)
        
        # Load example data
        pathnames = cl.import_example(data='flowers')
        
        # Read image according the preprocessing steps
        img = cl.imread(pathnames[10], dim=(128,128))
        
        # Extract HOG features
        img_hog = cl.extract_hog(img, pixels_per_cell=(8,8), orientations=8, flatten=False)
        
        plt.figure()
        fig,axs=plt.subplots(1,2, figsize=(15,10))
        axs[0].imshow(img.reshape(128,128,3))
        axs[0].axis('off')
        axs[0].set_title('Preprocessed image', fontsize=12)
        axs[1].imshow(img_hog, cmap='gray')
        # axs[1].imshow(img_hog.reshape(dim), cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('HOG', fontsize=12)

