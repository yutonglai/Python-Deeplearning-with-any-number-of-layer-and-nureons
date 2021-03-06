# Python-Deeplearning-with-any-number-of-layer-and-nureons


#### Head
    %pylab inline
    import numpy as np
    from scipy import optimize
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
 
 #### Defining the class
#### New complete class
    class NeuralNetwork(object):
        def __init__(self, HiddenLayer=0, X=0, Lambda=0, LearningRate=0, Activation='Sigmoid', a=0.5):  

            ## Define Hyperparameters
            self.LayerSize = np.concatenate(([X.shape[1]], HiddenLayer, [1]))
            # Learning Rate
            self.LearningRate = LearningRate
            # Regularization Parameter
            self.Lambda = Lambda
            # Activation Function
            self.Activation = Activation
            # parameter for ReLU
            self.a = a

            ## Array Initialization
            self.W     = [np.random.randn(self.LayerSize[0], self.LayerSize[1])]
            self.dJdW  = [np.random.randn(self.LayerSize[0], self.LayerSize[1])]
            self.alpha = [np.zeros((X.shape[0], self.LayerSize[0]))]
            self.z     = [np.zeros((X.shape[0], self.LayerSize[0]))]
            self.delta = [np.zeros((X.shape[0], self.LayerSize[0]))]              
            self.W_end = np.zeros(len(self.LayerSize)) 

            for i in range(len(self.LayerSize)-1):
                self.W.append(np.random.randn(self.LayerSize[i], self.LayerSize[i+1]))
                self.dJdW.append(np.random.randn(self.LayerSize[i], self.LayerSize[i+1]))
                self.alpha.append(np.zeros((X.shape[0], self.LayerSize[i+1])))
                self.z.append(np.zeros((X.shape[0], self.LayerSize[i+1])))
                self.delta.append(np.zeros((X.shape[0], self.LayerSize[i+1])))


        def activationFunction(self, z):
            # Apply sigmoid activation function to scalar, vector, or matrix
            if self.Activation == 'Sigmoid':
                return 1/(1+np.exp(-z))
            if self.Activation == 'ReLU':
                z[z<0] = z[z<0]*(self.a)
                return z
            if self.Activation == 'TanH':
                return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))


        def ActivationDerivative(self, z):
            # Gradient of sigmoid
            if self.Activation == 'Sigmoid':
                return np.exp(-z)/((1+np.exp(-z))**2)
            if self.Activation == 'ReLU':
                z[z>=0] = 1
                z[z<0] = self.a
                return -z
            if self.Activation == 'TanH':
                TanH = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
                return 1-TanH**2


        def setParams(self):
            # Set W1 and W2 using single parameter vector:  
            self.W_end[1] = self.LayerSize[0]*self.LayerSize[1]
            for i in range(len(self.LayerSize)-2):
                self.W_end[i+2] = self.W_end[i+1] + self.LayerSize[i+1]*self.LayerSize[i+2]
                self.W[i+2]     = np.reshape(self.params[int(self.W_end[i+1]):int(self.W_end[i+2])], (self.LayerSize[i+1], self.LayerSize[i+2]))
                self.W[i+2]     = self.W[i+2] - self.LearningRate*self.dJdW[i+2]
            return self.W


        def getParams(self):
            #Get W1 and W2 Rolled into vector:
            self.params = self.W[1].ravel()
            for i in range(len(self.LayerSize)-2):
                self.params = np.concatenate((self.params, self.W[i+2].ravel()))
            return self.params


        def getdJdW(self):
            #Get W1 and W2 Rolled into vector:
            self.paramsdJdW = self.dJdW[1].ravel()
            for i in range(len(self.LayerSize)-2):
                self.paramsdJdW = np.concatenate((self.paramsdJdW, self.dJdW[i+2].ravel()))
            return self.paramsdJdW


        def forwardPropagation(self, X):
            # Propogate inputs though network    
            self.alpha[0] = X
            for i in range(len(self.LayerSize)-1):
                self.z[i+1] = np.dot(self.alpha[i], self.W[i+1])
                self.alpha[i+1] = self.activationFunction(self.z[i+1])
            return self.alpha[-1]


        def costFunction(self, X, y):
            # Compute cost for given X,y, use weights already stored in class.
            yHat = self.forwardPropagation(X)
            J = 0.5*sum((y-yHat)**2)/X.shape[0]# + (self.Lambda/2)*(np.sum(self.W[1]**2)+np.sum(self.W[2]**2))
            # J = sum(-y*log(yHat) - (1-y)*log(1-yHat))/X.shape[0] + (self.Lambda/2)*(np.sum(self.W[1]**2)+np.sum(self.W[2]**2))
            return J


        def costFunctionDerivative(self, X, y):
            # Compute derivative with respect to W for given X and y:
            yHat = self.forwardPropagation(X)      
            #self.delta[-1] = np.multiply(-y/yHat+(1-y)/(1-yHat), np.exp(-self.z[-1])/((1+np.exp(-self.z[-1]))**2))
            #self.delta[-1] = np.multiply(-y/yHat+(1-y)/(1-yHat), self.Derivative(self.z[-1]))
            #self.delta[-1] = np.multiply(-(y-yHat), self.Derivative(self.z[-1]))
            self.delta[-1] = np.multiply(-(y-yHat), np.exp(-self.z[-1])/((1+np.exp(-self.z[-1]))**2))
            self.dJdW[-1]  = np.dot(self.alpha[-2].T, self.delta[-1])/X.shape[0] + self.Lambda*self.W[-1]
            for i in range(len(self.LayerSize)-2):
                self.delta[-i-2] = np.dot(self.delta[-i-1],   self.W[-i-1].T)*self.ActivationDerivative(self.z[-i-2])
                self.dJdW[-i-2]  = np.dot(self.alpha[-i-3].T, self.delta[-i-2])/X.shape[0] + self.Lambda*self.W[-i-2]     


               
#### Initialization
    trainx = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
    trainy = np.array(([75],  [82],  [93],   [70]),    dtype=float)

    #Testing Data:
    testx  = np.array(([4,5.5], [4.5,1], [9,2.5], [6,2]), dtype=float)
    testy  = np.array(([70],    [89],    [85],    [75]),  dtype=float)

    #Normalize:
    trainx = trainx/np.amax(trainx, axis=0)
    trainy = trainy/100
    #Normalize by max of training data:
    testx  = testx/np.amax(testx, axis=0)
    testy  = testy/100

    #Hyper-parameters
    HiddenLayer  = [10,10,10,10]
    Lambda       = 0.0004
    LearningRate = 2
    Iteration    = 10000


#### Implement
    #Train network with new data:
    NN    = NeuralNetwork(X=trainx, HiddenLayer=HiddenLayer, Lambda=Lambda,LearningRate=LearningRate)
    yHat  = NN.forwardPropagation(trainx)
    score = np.zeros((Iteration,len(trainx)))
    cost  = np.zeros(Iteration)
    testcost = np.zeros(Iteration)
    W = np.zeros((Iteration, len(NN.getParams())))
    for i in range(Iteration):
        W[i,:]      = NN.getParams()
        NN.costFunctionPrime(trainx, trainy)
        NN.setParams()
        yHat        = NN.forwardPropagation(trainx)
        testyHat    = NN.forwardPropagation(testx)
        score[i,]   = yHat.T
        cost[i]     = NN.costFunction(trainx, trainy)
        testcost[i] = NN.costFunction(testx, testy)


    print('trainy\n' + str(trainy.T))
    yHat = NN.forwardPropagation(trainx)
    print(np.round(yHat.T,2))

    print('testy\n' + str(testy.T))
    testyHat = NN.forwardPropagation(testx)
    print(np.round(testyHat.T,2))

    # 2-D plots
    plt.figure(figsize=(20,20))
    subplot(3,1,1)
    plot(score);grid(1)
    plt.legend(labels=['1','2','3','4'])
    subplot(3,1,2)
    plot(cost);grid(1)
    plot(testcost)
    subplot(3,1,3)
    plot(W);grid(1)
    W.shape

![2d](https://user-images.githubusercontent.com/46899273/53830341-d62cd100-3f47-11e9-9833-de8dce323a66.png)

#### 3-D plot:
    #%matplotlib qt

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')

    #Test network for various combinations of sleep/study:
    hoursSleep = linspace(0, 10,  100)
    hoursStudy = linspace(0,  5,  100)

    #Normalize data (same way training data way normalized)
    hoursSleepNorm = hoursSleep/10
    hoursStudyNorm = hoursStudy/5

    #Create 2D versions of input for plotting
    a, b = meshgrid(hoursSleepNorm, hoursStudyNorm)

    #Join into a single input matrix
    allInputs = np.zeros((a.size, 2))
    allInputs[:,0] = a.ravel()
    allInputs[:,1] = b.ravel()
    allOutputs = NN.forwardPropagation(allInputs)

    #Contour PLot:
    yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
    xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

    CS = contour(xx, yy, 100*allOutputs.reshape(100,100))
    clabel(CS, inline=1, fontsize=10)
    xlabel('Hours Sleep')
    ylabel('Hours Study')

    #Scatter training example:
    #plt.figure(figsize=(20,20))
    ax.scatter(10*testx[:,0], 5*testx[:,1], 100*testy, c='k', alpha=1, s=30)
    ax.scatter(10*trainx[:,0], 5*trainx[:,1], 100*trainy, c='r', alpha=1, s=30)
    surf = ax.plot_surface(xx, yy, \
                           100*allOutputs.reshape(100,100), \
                           cmap=cm.jet, \
                           alpha=0.5)

    ax.set_xlabel('Hours Sleep')
    ax.set_ylabel('Hours Study')
    ax.set_zlabel('Test  Score')
![3d](https://user-images.githubusercontent.com/46899273/53830444-112f0480-3f48-11e9-9844-4fc382080975.png)

