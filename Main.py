
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Delaunay import FaceSwap_Delaunay
from TPS import FaceSwap_TPS
class Main:
    def __init__(self):
        FaceSwap_Delaunay().main()
    #    FaceSwap_TPS().main()

Main()