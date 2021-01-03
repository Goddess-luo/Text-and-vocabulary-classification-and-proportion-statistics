#!/usr/bin/env python3
'''
Question answer:
In this assignment, my task is to train the train.JSON data set. I use LSTM in combination
with the full connection layer for training. In terms of the activation function, I tried
tanh and Relu, and I chose relu. On the optimizer side, I replaced SGD with an Adam optimizer
for accelerated training.

The first is pre-processing. Here I try to use stopWord in Sklearn, then filter the data
through the filter in Python and strip to remove punctuation marks, so as to improve the
accuracy. In setting up the network, I first use LSTM to Encoder sentence vectors, then get
a vector with 100 dimensions, and then get a vector with 64 dimensions through full connection.
Finally, according to the title requirements, it is divided into two Decoder layers of 64-> 32
-> 1 and 64-> 32-> 5, with the final output of two results.

Rating is a dichotomous task, so it enters BCELoss for loss calculation after sigmoid layer.
Category is a multi-classification task. Therefore, NLLLoss is calculated by softmax layer and
then added to the two losses respectively to obtain the total loss.
'''
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

# import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
import sklearn
import string

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    try:
        stopword = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
    except AttributeError as e:
        stopword = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

    sample = list(filter(lambda x : x not in stopword, sample))
    sample = list(map(lambda x: x.strip(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), sample))
    return sample
def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    return (ratingOutput >= 0.5).long(), categoryOutput.argmax(dim=1)

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(input_size=50, hidden_size=100, num_layers=2, batch_first=True, dropout=0.7)
        self.fc = tnn.Linear(in_features=100, out_features=64)
        self.relu = tnn.Tanh()
        self.fc_rateing = tnn.Linear(in_features=64, out_features=32)
        self.out_rating = tnn.Linear(in_features=32, out_features=1)
        self.fc_category = tnn.Linear(in_features=64, out_features=32)
        self.out_category = tnn.Linear(in_features=32, out_features=5)
        self.softmax = tnn.LogSoftmax(dim=1)
        self.sigmoid = tnn.Sigmoid()

    def forward(self, input, length):
        # x = self.conv(input.permute(0, 2, 1))
        o, (h, c) = self.lstm(input)
        x = o[:, -1, :]
        x = self.relu(self.fc(x))
        rating = self.sigmoid(self.out_rating(self.relu(self.fc_rateing(x))))
        category = self.softmax(self.out_category(self.relu(self.fc_category(x))))
        return rating, category


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.crossentropy = tnn.NLLLoss()
        self.bce = tnn.BCELoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        return self.bce(input=ratingOutput, target=ratingTarget.float()) + self.crossentropy(input=categoryOutput, target=categoryTarget)

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.001)
