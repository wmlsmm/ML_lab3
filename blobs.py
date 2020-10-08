import labfuns as lf
import lab3 as lab
import numpy as np

X, y = lf.genBlobs()
mu, sigma  = lab.mlParams(X, y)

#print("mu", mu, "\n\nsigma", sigma)
#print(lab.computePrior(y))
lf.plotGaussian(X, y, mu, sigma)

#clf = lab.BayesClassifier()
#clf = clf.trainClassifier(X, y)
# print("Classified ", clf.classify(np.array([[2,4],\
#                                             [2,0],\
#                                             [8,-3],\
#                                             [-2,2],\
#                                             [-2,8]])))


