# Linear model 
import numpy as np
from sklearn.cross_validation import StratifiedKFold

class Linear_classifier():
    def __init__(self,all_data, all_labels, classNr, lr=8e-1, reg=2e-3, nr_iters=100):
        self.Nr_features = all_data.shape[1] # 28x28 = 784 for mnist
        self.Nr_classes = classNr #Number of classes assuming class index starts from 0
        self.W = 0.01 * np.random.randn(self.Nr_features,self.Nr_classes)
        self.bias = np.zeros((1,self.Nr_classes))
        self.data = all_data
        self.labels = all_labels
        self.regularization_param = reg
        self.learning_rate = lr
        self.Nr_iterations = nr_iters
        self.loss=[]

        #For simplicity we will take the batch size to be the same as number of examples
        #self.batch_size = all_data.shape[0]

    def train_linear_classifier(self,X_train, y_train):
        batch_size = X_train.shape[0]
        # gradient descent loop
        for i in range(self.Nr_iterations):

            # evaluate class scores, [N x K]
            scores = np.dot(X_train, self.W) + self.bias
            #print("score values")
            #print(scores[:10])
            #print("length:",len(scores))

            # compute the class probabilities
            exp_scores = np.exp(scores)
            # print("score values")
            # print(exp_scores[:10])
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]


            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(batch_size),y_train])
            data_loss = np.sum(corect_logprobs)/batch_size
            #mi pare stia facendo L2 Regularization
            reg_loss = 0.5*self.regularization_param*np.sum(self.W*self.W)
            loss = data_loss + reg_loss

            self.loss.append(loss)

            if i % 50 == 0:
                print("iteration:", i , " loss:",loss)

            # compute the gradient on scores
            dscores = probs
            dscores[range(batch_size),y_train] -= 1
            dscores /= batch_size

            # backpropate the gradient to the parameters (W,b)
            dW = np.dot(X_train.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

            dW += self.regularization_param*self.W # regularization gradient

            # perform a parameter update
            self.W += -self.learning_rate * dW
            self.bias += -self.learning_rate * db

    def test_linear_classifier(self, X_data, y_data):
        # Post-training: evaluate validation set accuracy
        scores = np.dot(X_data, self.W) + self.bias
        predicted_class = np.argmax(scores, axis=1)
        test_accuracy = (np.mean(predicted_class == y_data))
        return test_accuracy
 

    # cross validation scheme
    def cross_validate_classifier(self, num_of_folds=10):
        # create stratified k folds of dataset for cross validation
        skf = StratifiedKFold(self.labels, n_folds=num_of_folds,random_state=0)
        # store predicted accuracies of each fold
        CV_pred_accuracies = []

        for train_index, valid_index in skf:
            X_train, X_valid = self.data[train_index], self.data[valid_index]
            y_train, y_valid = self.labels[train_index],self.labels[valid_index]
            self.train_linear_classifier(X_train, y_train)
            test_accuracy = self.test_linear_classifier(X_valid, y_valid)
            list.append(CV_pred_accuracies,test_accuracy)
            print("cumulative CV accuracy: ", np.mean(CV_pred_accuracies),"\n")
              