#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    W = np.ones([Npts, 1])
    for i, k  in enumerate(classes):
        idx = labels == k
        # without W
        prior[i] = np.sum(idx) / Npts
        # with W
        prior[i] = np.sum(W[idx]) / np.sum(W)
    # ==========================
    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points    (x1,y1,z1,...d),(x2,y2,z2,...)
#     labels - N vector of class labels             1               2/3
#          W - N x 1 vector of weights
# out:    mu - C x d matrix of class means (mu[i] = class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] = class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)     # array([1, 2, 3, 4])
    Nclasses = np.size(classes)     # 4

    if W is None:
        W = np.ones((Npts,1))/float(Npts)   # (1 1 1 1 1 ..... N)' / N = (1/N 1/N ...)'

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================

    for i, k in enumerate(classes):
        idx = labels == k
        xlc = X[idx,:]
        wlc = W[idx,:]
        mu[i, :] = np.sum(xlc, axis=0) / xlc.shape[0]
        sigma[i,:,:] = np.diag(np.sum(np.square( xlc-mu[i,:] ), axis=0) / xlc.shape[0])
        #print("Without w: \nmu: ", mu, "\nsigma: ",sigma)
        mu[i,:] = np.sum(wlc * xlc, axis = 0) / np.sum(wlc)
        sigma[i,:,:] = np.diag(np.sum(wlc * np.square( xlc-mu[i,:] ), axis=0) / np.sum(wlc))
        #print("With w: \nmu: ", mu, "\nsigma: ", sigma)
    # ==========================
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for idx_star in range (Npts):
        for idx_k in range(Nclasses):
            logProb[idx_k, idx_star] = -0.5 * np.log(np.linalg.det(sigma[idx_k])) - \
                                       0.5 * (X[idx_star] - mu[idx_k]) @ (np.linalg.inv(sigma[idx_k])) @ np.transpose((X[idx_star] - mu[idx_k])) \
                                       + np.log(prior[idx_k])
    # ==========================

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        #print(rtn.mu, rtn.sigma)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)



# ## Boosting functions to implement

# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        # 1.
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))
        
        # do classification for each point
        vote = classifiers[-1].classify(X)
        #print("vote: ", vote)
        #print("labels: ", labels.shape[0], labels)
        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        # 2.

        delta = np.zeros([Npts,1])
        for i in range(Npts):
            if vote[i] == labels[i]:
                delta[i] = 1
        error = np.sum(wCur * (np.ones([Npts,1])-delta) )
        #print("error: ", error)
        # 3.
        alpha = 0.5*(np.log(1-error) - np.log(error))
        alphas.append(alpha) # you will need to append the new alpha
        # 4.
        wNew = wCur
        for i in range(wCur.shape[0]):
            if delta[i] == 1:
                wNew[i] = (wCur[i] / 1 ) * np.exp(-alpha)
            else:
                wNew[i] = (wCur[i] / 1 ) * np.exp(alpha)
        Z = np.sum(wNew)
        wNew = wNew / Z

        '''
        #OVO NIJE MOJ KOD ALI DAJE BOLJE NEGO MOJ GORE
        #MOJ:classification accuracy  92.8 with standard deviation 7.12
        #   neki triali ispod 90
        #OVAJ: mean classification accuracy  93.9 with standard deviation 6.73
        #       nijedan trial ispod 90, svi iznad 93 (svi kazu u wapp grupi da tako treba bit, NITKO nema ispod 93)
        #CAN YOU SPOT THE DIFFERENCE? (mozda u error?)
        weightsum = np.sum(wCur)

        # Adaboost step (2)
        error = weightsum
        correctvote = np.where(vote == labels)[0]
        for i in correctvote:
            error -= wCur[i]

            # Adaboost step (3)
        curAlpha = (np.log(1 - error) - np.log(error)) / 2  # Compute new alpha
        alphas.append(curAlpha)  # you will need to append the new alpha

        # Adaboost step (4)
        falsevote = np.where(vote != labels)[0]
        wOld = wCur

        for i in correctvote:
            wCur[i] = wOld[i] * np.exp(-curAlpha)
        for i in falsevote:
            wCur[i] = wOld[i] * np.exp(curAlpha)
        wCur /= np.sum(wCur)
        '''
        # ==========================

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        # print("classifyBoost votes: ", votes)
        # print("classifyBoost alphas: ", alphas)
        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for i, classifier in enumerate(classifiers):
            vote = classifier.classify(X)
            for j in range(Npts):
                votes[j][vote[j]] += alphas[i]
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.

# ==============ASSIGNMENT 3 ======= Test the Maximum Likelihood estimates
# testClassifier(BayesClassifier(), dataset='iris', split=0.7)
# plotBoundary(BayesClassifier(), dataset='iris',split=0.7)
# testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
# plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)

# ============== ASSIGNMENT 5=======BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7))
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)

'''
# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])
'''
