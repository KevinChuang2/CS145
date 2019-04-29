# =======================================================================
from KMeans import KMeans
from DataPoints import DataPoints
import random
import math
# =======================================================================
class DBSCAN:
    # -------------------------------------------------------------------
    def __init__(self):
        self.e = 0.0
        self.minPts = 3
        self.noOfLabels = 0
    # -------------------------------------------------------------------
    def main(self, args):
        seed = 71
        print("For dataset1")
        dataSet = KMeans.readDataSet("dataset1.txt")
        random.Random(seed).shuffle(dataSet)
        self.noOfLabels = DataPoints.getNoOFLabels(dataSet)
        self.e = self.getEpsilon(dataSet)
        print(("Esp :" + str(self.e)))
        self.dbscan(dataSet)

        print("\nFor dataset2")
        dataSet = KMeans.readDataSet("dataset2.txt")
        random.Random(seed).shuffle(dataSet)
        self.noOfLabels = DataPoints.getNoOFLabels(dataSet)
        self.e = self.getEpsilon(dataSet)
        print(("Esp :" + str(self.e)))
        self.dbscan(dataSet)

        print("\nFor dataset3")
        dataSet = KMeans.readDataSet("dataset3.txt")
        random.Random(seed).shuffle(dataSet)
        self.noOfLabels = DataPoints.getNoOFLabels(dataSet)
        self.e = self.getEpsilon(dataSet)
        print(("Esp :" + str(self.e)))
        self.dbscan(dataSet)
    # -------------------------------------------------------------------
    def getEpsilon(self, dataSet):
        distances = []
        sumOfDist = 0.0
        for i in range(len(dataSet)):
            point = dataSet[i]
            dist = 0.0
            for j in range(len(dataSet)):
                if i == j:
                    continue
                pt = dataSet[j]
                dist = self.getEuclideanDist(point.x, point.y, pt.x, pt.y)
                distances.append(dist)

            distances.sort()
            sumOfDist += distances[7]
            distances = []
        return sumOfDist/len(dataSet)
    # -------------------------------------------------------------------
    def dbscan(self, dataSet):
        clusters = []
        visited = set()
        noise = set()

        # Iterate over data points
        for i in range(len(dataSet)):
            point = dataSet[i]
            if point in visited:
                continue
            visited.add(point)
            N = []
            minPtsNeighbours = 0

            # check which point satisfies minPts condition 
            for j in range(len(dataSet)):
                if i==j:
                    continue
                pt = dataSet[j]
                dist = self.getEuclideanDist(point.x, point.y, pt.x, pt.y)
                if dist <= self.e:
                    minPtsNeighbours += 1
                    N.append(pt)

            if minPtsNeighbours >= self.minPts:
                cluster = set()
                cluster.add(point)
                point.isAssignedToCluster = True

                j = 0
                while j < len(N):
                    point1 = N[j]
                    minPtsNeighbours1 = 0
                    N1 = []
                    if not point1 in visited:
                        visited.add(point1)
                        for l in range(len(dataSet)):
                            pt = dataSet[l]
                            dist = self.getEuclideanDist(point1.x, point1.y, pt.x, pt.y)
                            if dist <= self.e:
                                minPtsNeighbours1 += 1
                                N1.append(pt)
                        if minPtsNeighbours1 >= self.minPts:
                            self.removeDuplicates(N, N1)
                        else:
                            N1 = []
                        if not point1.isAssignedToCluster:
                            cluster.add(point1)
                    # Add point1 is not yet member of any other cluster then add it to cluster
                    # Hint: use self.isAssignedToCluster function to check if a point is assigned to any clusters
        # ****************Please Fill Missing Lines Here*****************                    
                    
                    
                    j += 1
                # add cluster to the list of clusters
                clusters.append(cluster)

            else:
                noise.add(point)

            N = []

        # List clusters
        print(("Number of clusters formed :" + str(len(clusters))))
        print(("Noise points :" + str(len(noise))))

        # Calculate purity
        maxLabelCluster = []
        for j in range(len(clusters)):
            maxLabelCluster.append(KMeans.getMaxClusterLabel(clusters[j]))
        purity = 0.0
        for j in range(len(clusters)):
            purity += maxLabelCluster[j]
        purity /= len(dataSet)
        print(("Purity is :" + str(purity)))

        nmiMatrix = DataPoints.getNMIMatrix(clusters, self.noOfLabels)
        nmi = DataPoints.calcNMI(nmiMatrix)
        print(("NMI :" + str(nmi)))

        DataPoints.writeToFile(noise, clusters, "DBSCAN_dataset3.csv")
    # -------------------------------------------------------------------
    def removeDuplicates(self, n, n1):
        for point in n1:
            isDup = False
            for point1 in n:
                if point1 == point:
                    isDup = True
            if not isDup:
                n.append(point)
    # -------------------------------------------------------------------
    def getEuclideanDist(self, x1, y1, x2, y2):
        dist = math.sqrt(pow((x2-x1), 2) + pow((y2-y1), 2))
        return dist
# =======================================================================
if __name__ == "__main__":
    d = DBSCAN()
    d.main(None)