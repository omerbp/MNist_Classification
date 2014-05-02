import read2Numpy2D as rn
import Perceptron as pr
import AdaBoost as ab

digits=[1,9] #which 2 digits you want to discriminate between? 
digit = digits[0] #the digit that will be labeles as 1
alpha=0.2 #controls the number of training iteration
Twl=30 #controls the adaboost's number of iterations
colnum=[i for i in xrange(0,0)]
rownum=[i for i in xrange(7,10)]

""" author: Omer Ben-Porat
this project deals with MNIST digits data base. it uses Scipy's containers to create linear classifiers 
and to classfity. it contains:
    1) read2Numpy2D - functions that read the database and reduces the data to smaller amount of features,
       in order to test some "weak learning" method as adaboost
    2) Perceptron - a Perceptron class and a multi-digit perceptron
    3) AdaBoost
    
I tried to use the notation from 'Understanding Machine Learning' by Shai Shalev-Shwartz and Shai Ben-David. 
hope you have fun using it"""


def main():  
    space = '\n'+"************************************************"+'\n'
    "ordinary perceptron"
    TrainAndTestBinPerceptron()
    print space
    
    "weak perceptron"
    TrainAndTestWeakBinPerceptron()
    print space
    
    "adaboost" 
    TrainAndTestAdaBoostWeakBinPerceptron()
    print space
    
    "bonus - multi digit perceptron"
    TrainAndTestMultiDigitsPerceptron()
    #TrainAndTestWeakMultiDigitPerceptron()
    
def TrainAndTestMultiDigitsPerceptron():
    digits = [i for i in xrange(0,10)]
    print "Running the Multi Digits Perceptron"
    print "the digits it shall descriminate between: "+digits.__str__()
    print "Extracting the Training Data"
    img,lbl = rn.getData(digits, "training")
    print "Training the Multi Digit Perceptron:"
    MulitDigitPerc = pr.MultiDigitPerceptron(digits,img,lbl,int(alpha*len(lbl)))
    print "Error on the Training data: "
    print str(pr.TestClassifier(img,lbl,MulitDigitPerc)*100.0)+"%"
    print "Extracting the Testing Data"
    img,lbl = rn.getData(digits, "testing")
    print "False Classification on the Testing data: "+str(pr.TestClassifier(img,lbl,MulitDigitPerc)*100.0)+"%"

def TrainAndTestWeakMultiDigitPerceptron():
    digits = [i for i in xrange(0,10)]
    print "Running the Weak Multi Digit Perceptron"
    fullimg,lbl = rn.getData(digits, "training")
    reducedList = rn.getReductionList(colnum, rownum)
    img = rn.ReduceSetDimension(reducedList, fullimg)
    perc = pr.MultiDigitPerceptron(digits,img,lbl,int(alpha*len(lbl)))
    print "Error on the Training data: "+str(pr.TestClassifier(img,lbl,perc)*100.0)+"%"
    print "(out of "+str(len(lbl))+" samples)"
    print "Extracting the Testing Data"
    fullimg,lbl = rn.getData(digits, "testing")  
    img = rn.ReduceSetDimension(reducedList, fullimg)
    print "Error on the Testing data: "+str(pr.TestClassifier(img,lbl,perc)*100.0)+"%"
    print "(out of "+str(len(lbl))+" samples)"

def TrainAndTestBinPerceptron():
    if len(digits)>2:
        print "digits should be 2 digits in the case of Binary Perceptron"
    print "Running the Binary Perceptron"
    print "the digits it shall descriminate between: "+str(digits[0])+","+str(digits[1])
    print "the label 1 will be given to the digit "+str(digits[0])
    img,lbl = rn.getData(digits, "training")
    binlbl=rn.BinaryLabels(lbl,digit)
    perc = pr.Perceptron(img,binlbl,int(alpha*len(binlbl)))
    print "Error on the Training data: "+str(pr.TestClassifier(img, binlbl,perc)*100.0)+"%"
    print "(out of "+str(len(binlbl))+" samples)"
    print "Extracting the Testing Data"
    img,lbl = rn.getData(digits, "testing")
    binlbl=rn.BinaryLabels(lbl,digit)
    print "Error on the Testing data: "+str(pr.TestClassifier(img,binlbl,perc)*100.0)+"%"
    print "(out of "+str(len(binlbl))+" samples)"
def TrainAndTestWeakBinPerceptron():
    if len(digits)>2:
        print "digits should be 2 digits in the case of Binary Perceptron"
    print "Running the Weak Binary Perceptron"
    print "the digits it shall descriminate between: "+str(digits[0])+","+str(digits[1])
    fullimg,lbl = rn.getData(digits, "training")
    binlbl=rn.BinaryLabels(lbl,digit)

    reducedList = rn.getReductionList(colnum, rownum)
    img = rn.ReduceSetDimension(reducedList, fullimg)
    perc = pr.Perceptron(img,binlbl,int(alpha*len(binlbl)))
    print "Error on the Training data: "+str(pr.TestClassifier(img, binlbl,perc)*100.0)+"%"
    print "(out of "+str(len(binlbl))+" samples)"
    print "Extracting the Testing Data"
    fullimg,lbl = rn.getData(digits, "testing")  
    img = rn.ReduceSetDimension(reducedList, fullimg)
    binlbl=rn.BinaryLabels(lbl,digit)
    print "Error on the Testing data: "+str(pr.TestClassifier(img,binlbl,perc)*100.0)+"%"
    print "(out of "+str(len(binlbl))+" samples)"
def TrainAndTestAdaBoostWeakBinPerceptron():
    if len(digits)>2:
        print "digits should be 2 digits in the case of Binary Perceptron"
    print "Running the AdaBoost with Weak Binary Perceptron"
    print "the digits it shall descriminate between: "+str(digits[0])+","+str(digits[1])
    print "the label 1 will be given to the digit "+str(digits[0])
    fullimg,lbl = rn.getData(digits, "training")
    binlbl=rn.BinaryLabels(lbl,digit)
    reducedList = rn.getReductionList(colnum, rownum)
    img = rn.ReduceSetDimension(reducedList, fullimg)
    adaboost = ab.AdaBoost(pr.Perceptron, img, binlbl,Twl,int(len(lbl)*alpha))
    print "Error on the Training data: "+str(pr.TestClassifier(img, binlbl,adaboost)*100.0)+"%"
    print "(out of "+str(len(binlbl))+" samples)"
    print "Extracting the Testing Data"
    fullimg,lbl = rn.getData(digits, "testing")  
    img = rn.ReduceSetDimension(reducedList, fullimg)
    binlbl=rn.BinaryLabels(lbl,digit)
    print "Error on the Testing data: "+str(pr.TestClassifier(img,binlbl,adaboost)*100.0)+"%"
    print "(out of "+str(len(binlbl))+" samples)"

if __name__ == '__main__':
    main()