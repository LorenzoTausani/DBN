# Linear model 
import numpy as np

class Linear_classifier():
    def __init__(self,X_train, y_train, classNr, lr, reg, nr_iters):
        self.Nr_features = X_train.shape[1] # 28x28 = 784 for mnist
        self.Nr_classes = classNr #Number of classes assuming class index starts from 0
        self.W = 0.01 * np.random.randn(self.Nr_features,self.Nr_classes)
        self.bias = np.zeros((1,self.Nr_classes))
        self.Training_data = X_train
        self.Training_labels = y_train
        self.regularization_param = reg
        self.learning_rate = lr
        self.Nr_iterations = nr_iters
        self.loss=[]

        #For simplicity we will take the batch size to be the same as number of examples
        self.batch_size = X_train.shape[0]

    def train_linear_classifier(self):
        # gradient descent loop
        for i in range(self.Nr_iterations):

            # evaluate class scores, [N x K]
            scores = np.dot(self.Training_data, self.W) + self.bias
            #print("score values")
            #print(scores[:10])
            #print("length:",len(scores))

            # compute the class probabilities
            exp_scores = np.exp(scores)
            # print("score values")
            # print(exp_scores[:10])
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]


            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(self.batch_size),self.Training_labels])
            data_loss = np.sum(corect_logprobs)/self.batch_size
            #mi pare stia facendo L2 Regularization
            reg_loss = 0.5*self.regularization_param*np.sum(self.W*self.W)
            loss = data_loss + reg_loss

            self.loss.append(loss)

            if i % 50 == 0:
                print("iteration:", i , " loss:",loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(self.batch_size),self.Training_labels] -= 1
            dscores /= self.batch_size

            # backpropate the gradient to the parameters (W,b)
            dW = np.dot(self.Training_data.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

            dW += self.regularization_param*self.W # regularization gradient

            # perform a parameter update
            W += -self.learning_rate * dW
            b += -self.learning_rate * db