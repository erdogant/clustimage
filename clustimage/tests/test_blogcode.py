from clustimage import Clustimage
import itertools as it
import numpy as np
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

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
