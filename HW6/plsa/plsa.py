from numpy import zeros, int8, log
from pylab import random
import sys
import jieba
import re
import time
import codecs

# WARNING: YOU DO NOT NEED TO CHANGE PREPROCESSING FUNCTION!
# segmentation, stopwords filtering and document-word matrix generating
# [return]:
# N : number of documents
# M : length of dictionary
# word2id : a map mapping terms to their corresponding ids
# id2word : a map mapping ids to terms
# X : document-word matrix, N*M, each line is the number of terms that show up in the document
def preprocessing(datasetFilePath, stopwordsFilePath):
    
    # read the stopwords file
    file = codecs.open(stopwordsFilePath, 'r', 'utf-8')
    stopwords = [line.strip() for line in file] 
    file.close()
    
    # read the documents
    file = codecs.open(datasetFilePath, 'r', 'utf-8')
    documents = [document.strip() for document in file] 
    file.close()

    # number of documents
    N = len(documents)

    wordCounts = [];
    word2id = {}
    id2word = {}
    currentId = 0;
    # generate the word2id and id2word maps and count the number of times of words showing up in documents
    for document in documents:
        segList = jieba.cut(document)
        wordCount = {}
        for word in segList:
            word = word.lower().strip()
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:               
                if word not in word2id.keys():
                    word2id[word] = currentId;
                    id2word[currentId] = word;
                    currentId += 1;
                if word in wordCount:
                    wordCount[word] += 1
                else:
                    wordCount[word] = 1
        wordCounts.append(wordCount);
    
    # length of dictionary
    M = len(word2id)  

    # generate the document-word matrix
    X = zeros([N, M], int8)
    for word in word2id.keys():
        j = word2id[word]
        for i in range(0, N):
            if word in wordCounts[i]:
                X[i, j] = wordCounts[i][word];    

    return N, M, word2id, id2word, X


def initializeParameters():
    for i in range(0, N):
        normalization = sum(theta[i, :])
        for j in range(0, K):
            theta[i, j] /= normalization;

    for i in range(0, K):
        normalization = sum(beta[i, :])
        for j in range(0, M):
            beta[i, j] /= normalization;


def EStep():
    for i in range(0, N):
        for j in range(0, M):
            ## ================== YOUR CODE HERE ==========================
            ###  for each word in each document, calculate its
            ###  conditional probability belonging to each topic (update p)

            # ============================================================

def MStep():
    for k in range(0, K):
        # ================== YOUR CODE HERE ==========================
        ###  Implement M step 1: given the conditional distribution
        ###  find the parameters that can maximize the expected likelihood (update beta)

        # ============================================================
        
    for i in range(0, N):
        # ================== YOUR CODE HERE ==========================
        ###  Implement M step 2: given the conditional distribution
        ###  find the parameters that can maximize the expected likelihood (update theta)

        # ============================================================


# calculate the log likelihood
def LogLikelihood():
    loglikelihood = 0
    for i in range(0, N):
        for j in range(0, M):
            # ================== YOUR CODE HERE ==========================
            ###  Calculate likelihood function

            # ============================================================
    return loglikelihood

# output the params of model and top words of topics to files
def output():
    # document-topic distribution
    file = codecs.open(docTopicDist,'w','utf-8')
    for i in range(0, N):
        tmp = ''
        for j in range(0, K):
            tmp += str(theta[i, j]) + ' '
        file.write(tmp + '\n')
    file.close()
    
    # topic-word distribution
    file = codecs.open(topicWordDist,'w','utf-8')
    for i in range(0, K):
        tmp = ''
        for j in range(0, M):
            tmp += str(beta[i, j]) + ' '
        file.write(tmp + '\n')
    file.close()
    
    # dictionary
    file = codecs.open(dictionary,'w','utf-8')
    for i in range(0, M):
        file.write(id2word[i] + '\n')
    file.close()
    
    # top words of each topic
    file = codecs.open(topicWords,'w','utf-8')
    for i in range(0, K):
        topicword = []
        ids = beta[i, :].argsort()
        for j in ids:
            topicword.insert(0, id2word[j])
        tmp = ''
        for word in topicword[0:min(topicWordsNum, len(topicword))]:
            tmp += word + ' '
        file.write(tmp + '\n')
    file.close()



#### Starting main program ####
datasetFilePath = 'dataset1.txt' # or 'dataset2.txt'
stopwordsFilePath = 'stopwords.dic'
K = 3   # number of topic
maxIteration = 20
threshold = 5
topicWordsNum = 10
docTopicDist = './output/docTopicDistribution.txt'
topicWordDist = './output/topicWordDistribution.txt'
dictionary = './output/dictionary.dic'
topicWords = './output/topics.txt'

# preprocessing
N, M, word2id, id2word, X = preprocessing(datasetFilePath, stopwordsFilePath)

# theta[i, j] : p(zj|di): 2-D matrix
theta = random([N, K])
# beta[i, j] : p(wj|zi): 2-D matrix
beta = random([K, M])
# p[i, j, k] : p(zk|di,wj): 3-D tensor
p = zeros([N, M, K])

initializeParameters() # normarlize 

# EM algorithm
oldLoglikelihood = 1
newLoglikelihood = 1
for i in range(0, maxIteration):
    EStep() #implement E step
    MStep() #implement M step
    newLoglikelihood = LogLikelihood()
    print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLoglikelihood))
    # you should see increasing loglikelihood
    if(oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < threshold):
        break
    oldLoglikelihood = newLoglikelihood

output()