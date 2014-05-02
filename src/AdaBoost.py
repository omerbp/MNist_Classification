import scipy as sp
import math

signTolabel = lambda x: 0 if x<=0 else 1 
labelTosign = lambda x: -1 if x==0 else 1 

class AdaBoost():
    """this AdaBoost is based on the AdaBoost 
    author: Omer Ben - Porat
    input: 0/1 Classifier,Training data,Binary Labels, number of adaboost iterations,
    number of iteration to train the input classifier.
    """
    def __init__(self,WeakLearner,training_data, \
                 bin_training_label,T,Twl):
        self.T = T
        self.weakLearner = WeakLearner
        numOfSamples = len(training_data)
        self.numOfFeatures = len(training_data[0])
        self.D = [1.0/numOfSamples for _ in xrange(0,numOfSamples)]
        self.w = [0.0 for _ in xrange(0,T)]
        self.ht = []
        for t in xrange(0,T):
            if t%10==0:
                print "getting weak learner number "+str(t)
            wl = self.weakLearner(training_data,bin_training_label,Twl,D=self.D)
            self.ht.append(wl)
            epsilon=0.00001 
            #we want to prevent the case where epsilon=0,
            #because the calculation of w relays on epsilon!=0 
            for i in xrange(0,numOfSamples):
                h_t_xi = self.ht[t].Classify(training_data[i])
                y_xi = bin_training_label[i]
                if h_t_xi != y_xi :
                    epsilon+=self.D[i]
            if epsilon>0.5:
                print "        warning: for the "+str(t)+" iteration got "+\
                    "{0:.2f}".format(epsilon)+" error the training set "
            self.w[t]=0.5*math.log(1.0/epsilon-1)
            for i in xrange(0,numOfSamples):
                yi = labelTosign(bin_training_label[i])
                hti =labelTosign(self.ht[t].Classify(training_data[i]))
                self.D[i]=self.D[i]*math.exp(-1.0*self.w[t]*yi*hti)
            self.D = sp.divide(self.D,sp.sum(self.D))

    def Classify(self,x):
        weightsum = 0.0     
        for i in xrange(self.T):
            predict=labelTosign((self.ht)[i].Classify(x))
            weightsum=weightsum+predict* self.w[i]
        return signTolabel(weightsum) 
            