# =======================================================================
import sys
import math
# =======================================================================
class DataPoints:
    # -------------------------------------------------------------------
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.isAssignedToCluster = False
    # -------------------------------------------------------------------
    def __key(self):
        return (self.label, self.x, self.y)
    # -------------------------------------------------------------------
    def __eq__(self, other):
        return self.__key() == other.__key()
    # -------------------------------------------------------------------
    def __hash__(self):
        return hash(self.__key())
    # -------------------------------------------------------------------
    @staticmethod
    def getMean(clusters, mean):
        for index, cluster in enumerate(clusters):
            sumX = 0
            sumY = 0
            num=0
            for point in cluster:
                sumX+=point.x
                sumY+=point.y
                num+=1
            mean[index][0] = sumX/num
            mean[index][1] = sumY/num
        # Initialize the mean for each cluster
        # ****************Please Fill Missing Lines Here*****************
    # -------------------------------------------------------------------
    @staticmethod
    def getStdDeviation(clusters, mean, stddev):
        for index, cluster in enumerate(clusters):
            sumX = 0
            sumY = 0
            num=0
            for point in cluster:
                sumX+=math.pow((point.x-mean[index][0]),2)
                sumY+=math.pow((point.y-mean[index][1]),2)
                num+=1
            stddev[index][0] = math.sqrt(sumX/num)
            stddev[index][1] = math.sqrt(sumY/num)
        # Initialize the std for each cluster
        # ****************Please Fill Missing Lines Here*****************
    # -------------------------------------------------------------------
    @staticmethod
    def getCovariance(clusters, mean, stddev, cov):
        for index,cluster in enumerate(clusters):
            sum = 0
            num = 0
            for point in cluster:
                sum+=((point.x-mean[index][0])* (point.y-mean[index][1]))
                num+=1
            covariance = sum/num
            cov[index] = [[math.pow(stddev[index][0],2), covariance ], 
                         [covariance, math.pow(stddev[index][1],2)]
                         ]
        # ****************Please Fill Missing Lines Here*****************
    # -------------------------------------------------------------------
    @staticmethod
    def getNMIMatrix(clusters, noOfLabels):
        nmiMatrix = [[0 for x in range(len(clusters) + 1)] for y in range(noOfLabels + 1)]
        clusterNo = 0
        for cluster in clusters:
            labelCounts = {}
            for point in cluster:
                if not point.label in labelCounts:
                    labelCounts[point.label] = 0
                labelCounts[point.label] += 1
            max = sys.maxsize
            labelNo = 0
            labelTotal = 0
            labelCounts_sorted = sorted(labelCounts.items(), key=lambda x: (x[1], x[0]), reverse=True)
            for label, val in labelCounts_sorted:
                nmiMatrix[label - 1][clusterNo] = labelCounts[label]
                labelTotal += labelCounts.get(label)
            nmiMatrix[noOfLabels][clusterNo] = labelTotal
            clusterNo += 1
            labelCounts.clear()

        # populate last col
        lastRowCol = 0
        for i in range(len(nmiMatrix) - 1):
            totalRow = 0
            for j in range(len(nmiMatrix[i]) - 1):
                totalRow += nmiMatrix[i][j]
            lastRowCol += totalRow
            nmiMatrix[i][len(clusters)] = totalRow
        nmiMatrix[noOfLabels][len(clusters)] = lastRowCol
        return nmiMatrix
    # -------------------------------------------------------------------
    @staticmethod
    def calcNMI(nmiMatrix):
        # calculate I
        row = len(nmiMatrix)
        col = len(nmiMatrix[0])
        N = nmiMatrix[row - 1][col - 1]
        I = 0.0
        HOmega = 0.0
        HC = 0.0
        print(nmiMatrix)
        for i in range(row - 1):
            for j in range(col - 1):
                logPart = (float(N) * nmiMatrix[i][j]) / (float(nmiMatrix[i][col - 1]) * nmiMatrix[row - 1][j])
                if logPart == 0.0:
                    continue
                I += (nmiMatrix[i][j] / float(N)) * math.log(float(logPart))
                logPart1 = nmiMatrix[row - 1][j] / float(N)
                if logPart1 == 0.0:
                    continue
                HC += nmiMatrix[row - 1][j] / float(N) * math.log(float(logPart1))
            HOmega += nmiMatrix[i][col - 1] / float(N) * math.log(nmiMatrix[i][col - 1] / float(N))

        return I / math.sqrt(HC * HOmega)
    # -------------------------------------------------------------------
    @staticmethod
    def getNoOFLabels(dataSet):
        labels = set()
        for point in dataSet:
            labels.add(point.label)
        return len(labels)
    # -------------------------------------------------------------------
    @staticmethod
    def writeToFile(noise, clusters, fileName):
        # write clusters to file for plotting
        f = open(fileName, 'w')
        for pt in noise:
            f.write(str(pt.x) + "," + str(pt.y) + ",0" + "\n")
        for w in range(len(clusters)):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str((w + 1)) + "\n")
        f.close()
# =======================================================================