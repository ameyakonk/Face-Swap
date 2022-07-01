import numpy as np
import cv2
from facePoints import FaceFeatureExtractor
import matplotlib.pyplot as plt
import math

class FaceSwap_TPS():

    def calc_U(self, r):
        U = r**2*np.log(r**2)
        if(math.isnan(U)): return 0
        return U

    def find_w(self, pts2, pts1_):
        lambda_ = 0.0000001
        p = pts2.shape[0]
        K = np.zeros((p, p))
        pts1_ = np.reshape(pts1_, (p, 1))

        print(np.linalg.norm(pts2[0] - pts2[1]))
        for i in range(p):
            for j in range(p):
                K[i][j] = self.calc_U(np.linalg.norm(pts2[i] - pts2[j]))
     
        P = np.column_stack((pts2, np.ones((p,1))))
        P_T = P.T
        P_T_0 = np.column_stack((P_T, np.zeros((3, 3))))
        R_1 = np.column_stack((K, P))
        R = np.row_stack((R_1, P_T_0))
        Identity = np.identity(p+3)
        Final = R + lambda_*Identity
        Final_Inv = np.linalg.inv(Final)
 
        pts1_ = np.row_stack((pts1_, np.zeros((3,1))))
        return np.dot(Final_Inv, pts1_)

    def tps(self, src, tar, ff_coord_src, ff_coord_tar):

        ff_coord_src = np.asarray(ff_coord_src)
        ff_coord_tar = np.asarray(ff_coord_tar)

        p = ff_coord_src.shape[0]
        
        r = cv2.boundingRect(np.float32(ff_coord_tar))

        mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
        warped_img = np.copy(mask)

        points2_t = []

        hull = cv2.convexHull(ff_coord_tar, False).reshape(1, -1, 2)[0]
      
        for i in range(len(hull)):
            points2_t.append(((hull[i][0]-r[0]),(hull[i][1]-r[1])))
        #     cv2.circle(mask, ((hull[i][0]-r[0]),(hull[i][1]-r[1])), 2, (255, 0, 0), -1)

        # cv2.imshow("1", mask)
        # cv2.waitKey(0)
        
        cv2.fillConvexPoly(mask, np.int32(points2_t), (1.0, 1.0, 1.0), 16, 0)
        
        w_x = self.find_w(ff_coord_tar, ff_coord_src[:, 0])
        w_y = self.find_w(ff_coord_tar, ff_coord_src[:, 1])

        a_x = [w_x[p], w_x[p+1], w_x[p+2]]
        a_y = [w_y[p], w_y[p+1], w_y[p+2]]
        
        for i in range(mask.shape[1]):
            for j in range(mask.shape[0]):

                sum_x = 0
                sum_y = 0
                n = r[0] + i
                m = r[1] + j
                pt = [n, m]
                for k in range(ff_coord_tar.shape[0]):
                    sum_x += w_x[k]*self.calc_U(np.linalg.norm(ff_coord_tar[k,:] - pt))
                    sum_y += w_y[k]*self.calc_U(np.linalg.norm(ff_coord_tar[k,:] - pt))
                
                x = int(a_x[2] + a_x[0]*n + a_x[1]*m + sum_x)
                y = int(a_y[2] + a_y[0]*n + a_y[1]*m + sum_y)

                x = min(max(x, 0), src.shape[1]-1)
                y = min(max(y, 0), src.shape[0]-1)
                warped_img[j, i] = src[y, x,:]
              
        tar[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = tar[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( (1.0, 1.0, 1.0) - mask )
        tar[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = tar[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] + warped_img
        cv2.imshow("1", tar)
        cv2.waitKey(0)

    def main(self):
        src= "Data/source.jpg"
        src= cv2.imread(src)
        ff_coord_src = FaceFeatureExtractor().execute(src.copy())

        tar= "Data/target.jpg"
        tar= cv2.imread(tar)
        ff_coord_tar = FaceFeatureExtractor().execute(tar.copy())

        self.tps(src, tar, ff_coord_src[0], ff_coord_tar[0])

    
    
    
