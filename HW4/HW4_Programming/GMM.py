# =======================================================================
from DataPoints import DataPoints
from KMeans import KMeans
import math
from scipy.stats import multivariate_normal
# =======================================================================
class GMM:
    # -------------------------------------------------------------------
    def __init__(self):
        self.dataSet = []
        self.K = 0
        self.mean = [[0.0 for x in range(2)] for y in range(3)]
        self.stdDev = [[0.0 for x in range(2)] for y in range(3)]
        self.coVariance = [[[0.0 for x in range(2)] for y in range(2)] for z in range(3)]
        self.W = None
        self.w = None
    # -------------------------------------------------------------------
    def main(self, args):
        print("For dataset1")
        self.dataSet = KMeans.readDataSet("dataset1.txt")
        self.K = DataPoints.getNoOFLabels(self.dataSet)
        self.W = [[0.0 for y in range(self.K)] for x in range(len(self.dataSet))]
        self.w = [0.0 for x in range(self.K)]
        self.GMM()

        print("\n\n\nFor dataset2")
        self.dataSet = KMeans.readDataSet("dataset2.txt")
        self.K = DataPoints.getNoOFLabels(self.dataSet)
        self.W = [[0.0 for y in range(self.K)] for x in range(len(self.dataSet))]
        self.w = [0.0 for x in range(self.K)]
        self.GMM()

        print("\n\n\nFor dataset3")
        self.dataSet = KMeans.readDataSet("dataset3.txt")
        self.K = DataPoints.getNoOFLabels(self.dataSet)
        self.W = [[0.0 for y in range(self.K)] for x in range(len(self.dataSet))]
        self.w = [0.0 for x in range(self.K)]
        self.GMM()
    # -------------------------------------------------------------------
    def GMM(self):
        clusters = []
        self.mean = [[0.0 for y in range(2)] for x in range(self.K)]
        self.stdDev = [[0.0 for y in range(2)] for x in range(self.K)]
        self.coVariance = [[[0.0 for z in range(2)] for y in range(2)] for x in range(self.K)]
        k = 0
        while k < self.K:
            cluster = set()
            clusters.append(cluster)
            k += 1

        # Initially randomly assign points to clusters
        i = 0
        for point in self.dataSet:
            clusters[i % self.K].add(point)
            i += 1

        for m in range(self.K):
            self.w[m] = 1.0 / self.K

        # Get Initial mean
        DataPoints.getMean(clusters, self.mean)
        DataPoints.getStdDeviation(clusters, self.mean, self.stdDev)
        DataPoints.getCovariance(clusters, self.mean, self.stdDev, self.coVariance)
        length = 0
        mle_old = 0.0
        mle_new = 0.0
        while True:
            mle_old = self.Likelihood()
            self.Estep()
            self.Mstep(clusters)
            length += 1
            mle_new = self.Likelihood()

            # convergence condition
            if abs(mle_new - mle_old) / abs(mle_old) < 0.000001:
                break

        print(("Number of Iterations = " + str(length)))
        print("\nAfter Calculations")
        print("Final mean = ")
        self.printArray(self.mean)
        print("\nFinal covariance = ")
        self.print3D(self.coVariance)

        # Assign points to cluster depending on max prob.
        for j in range(self.K):
            clusters[j] = set()

        i = 0
        for point in self.dataSet:
            index = -1
            prob = 0.0
            for j in range(self.K):
                if self.W[i][j] > prob:
                    index = j
                    prob = self.W[i][j]
            temp = clusters[index]
            temp.add(point)
            i += 1

        # Calculate purity
        maxLabelCluster = [0 for x in range(self.K)]
        for j in range(self.K):
            maxLabelCluster[j] = KMeans.getMaxClusterLabel(clusters[j])
        purity = 0.0
        for j in range(self.K):
            purity += maxLabelCluster[j]
        purity = purity / float(len(self.dataSet))
        print(("Purity is :" + str(purity)))

        noOfLabels = DataPoints.getNoOFLabels(self.dataSet)
        nmiMatrix = DataPoints.getNMIMatrix(clusters, noOfLabels)
        nmi = DataPoints.calcNMI(nmiMatrix)
        print(("NMI :" + str(nmi)))

        # write clusters to file for plotting
        f = open("GMM.csv", 'w')
        for w in range(self.K):
            print(("Cluster " + str(w) + " size :" + str(len(clusters[w]))))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str(w) + "\n")
        f.close()
    # -------------------------------------------------------------------
    def Estep(self):
        for i in range(len(self.dataSet)):
            denominator = 0.0
            for j in range(self.K):
                gaussian = multivariate_normal(self.mean[j], self.coVariance[j])
                numerator = self.w[j] * gaussian.pdf([self.dataSet[i].x, self.dataSet[i].y])
                self.W[i][j] = numerator
                denominator += numerator
            for j in range(self.K):
                self.W[i][j] /=denominator
            
            # normalize W[i][j] into probabilities
            # ****************Please Fill Missing Lines Here*****************
    # -------------------------------------------------------------------
    def Mstep(self, clusters):
        for j in range(self.K):
            denominator = 0.0
            numerator = 0.0
            numerator1 = 0.0
            cov_xy = 0.0
            updatedMean1 = 0.0
            updatedMean2 = 0.0
            for i in range(len(self.dataSet)):
                denominator += self.W[i][j]
                numerator += self.W[i][j] * pow((self.dataSet[i].x - self.mean[j][0]), 2)
                numerator1 += self.W[i][j] * pow((self.dataSet[i].y - self.mean[j][1]), 2)
                cov_xy += self.W[i][j] * (self.dataSet[i].x - self.mean[j][0]) * (self.dataSet[i].y - self.mean[j][1])
                # ****************Please Fill Missing Lines Here*****************

                updatedMean1 += self.W[i][j] * self.dataSet[i].x
                updatedMean2 += self.W[i][j] * self.dataSet[i].y

            self.stdDev[j][0] = numerator / denominator
            self.stdDev[j][1] = numerator1 / denominator
            # update w[j]
            #for j in range(self.K):
            self.w[j] = denominator/ len(self.dataSet)
            # ****************Please Fill Missing Lines Here*****************
            # update mean
            self.mean[j][0] = updatedMean1 / denominator
            self.mean[j][1] = updatedMean2 / denominator

            # update covariance matrix
            self.coVariance[j][0][0] = self.stdDev[j][0]
            self.coVariance[j][1][1] = self.stdDev[j][1]
            self.coVariance[j][0][1] = self.coVariance[j][1][0] = cov_xy / denominator
    # -------------------------------------------------------------------
    def Likelihood(self):
        likelihood = 0.0
        for i in range(len(self.dataSet)):
            numerator = 0.0
            for j in range(self.K):
                gaussian = multivariate_normal(self.mean[j], self.coVariance[j])
                numerator += self.w[j] * gaussian.pdf([self.dataSet[i].x, self.dataSet[i].y])
            likelihood += math.log(numerator)
        return likelihood
    # -------------------------------------------------------------------
    def printArray(self, mat):
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                print((str(mat[i][j]) + " "), end=' ')
            print("")
    # -------------------------------------------------------------------
    def print3D(self, mat):
        for i in range(len(mat)):
            print(("For Cluster : " + str((i + 1))))
            for j in range(len(mat[i])):
                for k in range(len(mat[i][j])):
                    print((str(mat[i][j][k]) + " "), end=' ')
                print("")
            print("")
    # -------------------------------------------------------------------
    # Helper function to plot points in Excel
    def plot(self):
        f = open("xcel.csv", 'w')
        for i in range(len(self.dataSet)):
            point = self.dataSet[i]
            label = point.label
            f.write(point.x + "," + point.y + "," + point.label + "\n")
        f.close()
# =======================================================================
if __name__ == "__main__":
    g = GMM()
    g.main(None)