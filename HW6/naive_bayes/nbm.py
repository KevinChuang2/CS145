import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import seaborn as sns; sns.set()
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
from sklearn.metrics import confusion_matrix

class NB_model():
    def __init__(self): 
        self.pi = {} # to store prior probability of each class 
        self.Pr_dict = None
        self.num_vocab = None
        self.num_classes = None
    
    def fit(self, train_data, train_label, vocab, if_use_smooth=True):
        # get prior probabilities
        self.num_vocab = len(vocab['index'].tolist())
        self.get_prior_prob(train_label)
        # ================== YOUR CODE HERE ==========================
        # Calculate probability of each word based on class 
        # Hint: Store each probability value in matrix or dict: self.Pr_dict[classID][wordID] or Pr_dict[wordID][classID])
        # Remember that there are possible NaN or 0 in Pr_dict matrix/dict. Use smooth method
        
        # ============================================================
        print("Training completed!")
    
    def predict(self, test_data):
        test_dict = test_data.to_dict() # change dataframe to dict
        new_dict = {}
        prediction = []
        
        for idx in range(len(test_dict['docIdx'])):
            docIdx = test_dict['docIdx'][idx]
            wordIdx = test_dict['wordIdx'][idx]
            count = test_dict['count'][idx]
            try: 
                new_dict[docIdx][wordIdx] = count 
            except:
                new_dict[test_dict['docIdx'][idx]] = {}
                new_dict[docIdx][wordIdx] = count
                ''
        for docIdx in range(1, len(new_dict)+1):
            score_dict = {}
            #Creating a probability row for each class
            for classIdx in range(1,self.num_classes+1):
                score_dict[classIdx] = 0
                # ================== YOUR CODE HERE ==========================
                ### Implement the score_dict for all classes for each document
                ### Remember to use log addtion rather than probability multiplication
                ### Remember to add prior probability, i.e. self.pi

                # ============================================================
            max_score = max(score_dict, key=score_dict.get)
            prediction.append(max_score)
        return prediction
                    
    
    def get_prior_prob(self,train_label, verbose=True):
        unique_class = list(set(train_label))
        self.num_classes = len(unique_class)
        total = len(train_label)
        for c in unique_class:
            # ================== YOUR CODE HERE ==========================
            ### calculate prior probability of each class ####
            ### Hint: store prior probability of each class in self.pi

            # ============================================================
        if verbose:
            print("Prior Probability of each class:")
            print("\n".join("{}: {}".format(k, v) for k, v in self.pi.items()))

########### Data processing ##########
# read train/test labels from files
train_label = pd.read_csv('./dataset/train.label',names=['t'])
train_label = train_label['t'].tolist()
test_label = pd.read_csv('./dataset/test.label', names=['t'])
test_label= test_label['t'].tolist()
# read train/test documents from files
train_data = open('./dataset/train.data')
df_train = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
test_data = open('./dataset/test.data')
df_test = pd.read_csv(test_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
# read vocab
vocab = open('./dataset/vocabulary.txt') 
vocab_df = pd.read_csv(vocab, names = ['word']) 
vocab_df = vocab_df.reset_index() 
vocab_df['index'] = vocab_df['index'].apply(lambda x: x+1) 

#Add label column to original df_train (for better implementation)
docIdx = df_train['docIdx'].values
i = 0
new_label = []
for index in range(len(docIdx)-1):
    new_label.append(train_label[i])
    if docIdx[index] != docIdx[index+1]:
        i += 1
new_label.append(train_label[i])
df_train['classIdx'] = new_label

# output the following head of df_train dataframe if program correctly runned
#		docIdx	wordIdx	count	classIdx
#	0	1		1		4		1
#	1	1		2		2		1
#	2	1		3		10		1
#	3	1		4		4		1
#	4	1		5		2		1
df_train['classIdx'] = new_label
print(df_train.head()) # you may comment this line if you get correct results


########### Start to Train your model ##########
nbm = NB_model()
nbm.fit(df_train, train_label, vocab_df)

# make predictions on train set to validate the model
predict_train_labels = nbm.predict(df_train)
train_acc = (np.array(train_label) == np.array(predict_train_labels)).mean()
print("Accuracy on training data by my implementation: {}".format(train_acc))

# make predictions on test data
predict_test_labels = nbm.predict(df_test)
test_acc = (np.array(test_label) == np.array(predict_test_labels)).mean()
print("Accuracy on training data by my implementation: {}".format(test_acc))

# plot classification matrix
mat = confusion_matrix(test_label, predict_test_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig('./output/nbm_mine.png')
