import numpy as np
import cv2
import matplotlib.pyplot as plt
from facePoints import FaceFeatureExtractor

class FaceSwap_Delaunay:
    
    def rect_contains(self, rect, point) :
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True

    def draw_delaunay(self, img, triangleList, delaunay_color, points) :
        
        size = img.shape
        r = (0, 0, size[1], size[0])
        finalList = []
        for t in triangleList :

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            pt = [pt1, pt2, pt3]
           
            if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3) :
                temp = []
                for data in pt:
                    for p in range(len(points)) :
                        if (np.abs(data[0] - points[p][0]) < 1 and np.abs(data[1] - points[p][1]) < 1):
                            temp.append(p)
                            break   

                if(len(temp) == 3):
                    finalList.append(temp)

                cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
        return finalList

    def drawTriangles(self, points, img, animate = True):
        size = img.shape
        rect = (0, 0, size[1], size[0])

        # Create an instance of Subdiv2D
        subdiv = cv2.Subdiv2D(rect);
        for p in points[0] :
            subdiv.insert(p)

        triangleList = subdiv.getTriangleList();
        # Show animation
        ff = []
        if animate :
            img_copy = img.copy()
            # Draw delaunay triangles
            ff = self.draw_delaunay( img_copy, triangleList, (255, 255, 255), points[0]);
            # cv2.imshow("Delaunay Triangulation", img_copy)
            # cv2.waitKey(1000)
            return ff

    def getMatchedPoints(self, tl_src, ff_coord_src, ff_coord_tar):
        fl_src = []
        fl_tar = []
        for data in tl_src:
            temp_src = []
            temp_tar = []
            for i in range(3):
                temp_src.append(ff_coord_src[data[i]])
                temp_tar.append(ff_coord_tar[data[i]])

            fl_src.append(temp_src)
            fl_tar.append(temp_tar)
        return fl_src, fl_tar

    def drawMarkers(self, fl_src, fl_tar, src_img, tar_img):
        
        
        for i in range(len(fl_src)):
            a = np.random.randint(255)
            b = np.random.randint(255)
            c = np.random.randint(255)
            
            color = (a, b, c)
            for j in range(3):
                cv2.circle(src_img, fl_src[i][j], 3, color, -1)     
                cv2.circle(tar_img, fl_tar[i][j], 3, color, -1)   
        plt.subplot(2, 1, 1)
        plt.imshow(src_img)
        plt.subplot(2, 1, 2)
        plt.imshow(tar_img)
        plt.show()

    def findTrianglecoordinates(self, pt, img1):
        m,n,r = img1.shape
        mask = np.zeros((m+2, n+2), np.uint8)
        for i in range(len(pt)): 
            contours1 = np.asarray(pt[i])
            new_img1 = img1.copy()
            cv2.fillPoly(new_img1, pts = [contours1], color =(255,255,255))
            cv2.imshow("1", new_img1)
            cv2.waitKey(1000)

    def warpTriangles(self, pt1, pt2, src_img, tar_img):
        r1 = cv2.boundingRect(pt1) 
        r2 = cv2.boundingRect(pt2)
        # Offset points by left top corner of the respective rectangles
        tri1Cropped = []
        tri2Cropped = []
            
        for i in range(3):
            tri1Cropped.append(((pt1[i][0] - r1[0]),(pt1[i][1] - r1[1])))
            tri2Cropped.append(((pt2[i][0] - r2[0]),(pt2[i][1] - r2[1])))

        # Crop input image
        img1Cropped = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        
        warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
        img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

        img2Cropped = img2Cropped * mask
            
        # Copy triangular region of the rectangular patch to the output image
        tar_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = tar_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
            
        tar_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = tar_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

        
    def faceSwap(self, fl_src, fl_tar, src_img, tar_img):  
        # self.findTrianglecoordinates(fl_tar, tar_img)
        # self.findTrianglecoordinates(fl_src, src_img)

        for i in range(len(fl_src)):
            src_pt = np.asarray(fl_src[i])
            tar_pt = np.asarray(fl_tar[i])
            self.warpTriangles(src_pt, tar_pt, src_img, tar_img)

        return tar_img

    def main(self):
        src= "Data/source.jpg"
        src= cv2.imread(src)
        ff_coord_src = FaceFeatureExtractor().execute(src.copy())
        tl_src = self.drawTriangles(ff_coord_src, src.copy())

        cap = cv2.VideoCapture('Data/Jimmy.mp4')

        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                tar= "Data/target.jpg"
                tar = frame
                tar= cv2.imread(tar)
                ff_coord_tar = FaceFeatureExtractor().execute(tar.copy())
             
                if(len(ff_coord_tar) > 0 ):
                    self.drawTriangles(ff_coord_tar, tar.copy(), animate = False)
                    
                    fl_src, fl_tar = self.getMatchedPoints(tl_src, ff_coord_src[0], ff_coord_tar[0])
                #    self.drawMarkers(fl_src, fl_tar, src.copy(), tar.copy())
                    op_img = self.faceSwap(fl_src, fl_tar, src.copy(), tar.copy())

                    ff_coord_tar = np.asarray(ff_coord_tar)
                    hull = cv2.convexHull(ff_coord_tar, False).reshape(1, -1, 2)[0]
                    mask = np.zeros(op_img.shape, dtype = op_img.dtype)

                    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))
                    r = cv2.boundingRect(np.float32(ff_coord_tar[0]))    
                    
                    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
                
                    output = cv2.seamlessClone(np.uint8(op_img), tar, mask, center, cv2.NORMAL_CLONE)
                    out.write(output)
                    cv2.imshow("final_img", output)
                    cv2.waitKey(10) 
                
                else:
                    cv2.imshow("final_img", frame)
                    cv2.waitKey(1) 
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()