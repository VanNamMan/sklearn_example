
"""
use mean shift --> auto clustering.
"""
print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

from libs.visionLib import *

def main(file_name):
    t0 = time.time()

    mat = cv2.imread(file_name,0)
    ret,binary = cv2.threshold(mat,100,255,cv2.THRESH_BINARY)
    _,cnts,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(" ### Len contour : %d ###"%len(cnts))

    mat = cv2.cvtColor(mat,cv2.COLOR_GRAY2BGR)

    for i in range(len(cnts)):

        centroid = getCentroid(cnts[i])

        # approxy contours
        length = cv2.arcLength(cnts[i],True)
        
        epsilon = 0.01*length
        if epsilon < 3:
            epsilon = 3

        approx = cv2.approxPolyDP(cnts[i],epsilon,True)
        approx = approx.reshape(len(approx),2)

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(approx, quantile=0.2, n_samples=len(approx))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(approx)

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        for p in cluster_centers:
            p = (int(p[0]),int(p[1]))

            cv2.circle(mat,p,5,(0,255,0),-1)

        # labels_unique = np.unique(labels)
        # n_clusters_ = len(labels_unique)

        # print("number of estimated clusters : %d" % n_clusters_)

        # # Plot result
        # import matplotlib.pyplot as plt
        # from itertools import cycle

        # plt.figure(1)
        # plt.clf()

        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(n_clusters_), colors):
        #     my_members = labels == k
        #     cluster_center = cluster_centers[k]
        #     plt.plot(approx[my_members, 0], approx[my_members, 1], col + '.')
        #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=14)
        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        # plt.show()



        print("-Contours[%d]"%i)
        print("\t*centroid : ",centroid, "in contour : ",list(centroid) in arrContour2ListPoints(approx),"\n")
        print("\t*length : %.2f , epsilon : %.2f\n"%(length,epsilon))
        print("\t*approx : ",approx.shape,"\n")
        print("\t*cluster : ",cluster_centers.shape,"\n")
        
        cv2.putText(mat,"%d"%i,centroid,cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
    #     cv2.circle(mat,centroid,5,(0,255,),-1)
    #     for p in approx:
    #         cv2.circle(mat,tuple(p),3,(0,0,255),-1)

    dt = time.time()-t0
    print("* total time : %.2f\n"%dt)

    cv2.imshow("", mat)
    k = cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    file_name = sys.argv[1]
    main(file_name)