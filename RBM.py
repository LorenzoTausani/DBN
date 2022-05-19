import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import sys
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Linear_model_tf import LinearClassifier
import os

BATCH_SIZE = 64

class RBM(nn.Module): #nn.Module: Base class for all neural network modules.
    '''
    This class defines all the functions needed for an BinaryRBN model
    where the visible and hidden units are both considered binary
    '''

    def __init__(self,
                visible_units=256,
                hidden_units = 64,
                k=2, #k dovrebbe essere il numero di step di gibbs sampling
                learning_rate=1e-5,
                learning_rate_decay = False,
                xavier_init = False,
                increase_to_cd_k = False,
                use_gpu = False
                ):
        '''
        Defines the model
        W:Wheights shape (visible_units,hidden_units)
        c:hidden unit bias shape (hidden_units , )
        b : visible unit bias shape(visisble_units ,)
        '''
        #https://www.programiz.com/python-programming/methods/built-in/super
        #https://www.youtube.com/watch?v=aBexJgZ6GjI
        #super mi permette di chiamare l'init method dalla superclasse

        #la scrittura super(RBM,self) sembra essere solo un residuo di python2
        #https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        super(RBM,self).__init__()
        self.desc = "RBM"

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.xavier_init = xavier_init
        self.increase_to_cd_k = increase_to_cd_k
        self.use_gpu = use_gpu
        self.batch_size = 16
        self.h_train_labels = []
        self.h_test_labels = []
        self.nr_train_epochs_done = 0
        self.nr_train_epochs_done_CLASSIFIER = 0
        self.RBM_train_loss=[]
        self.RBM_train_loss_std=[]
        self.CLASSIFIER_train_loss=[]



        # Initialization
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Device = DEVICE

        if not self.xavier_init:
            self.W = torch.randn(self.visible_units,self.hidden_units) * 0.01 #weights
            #https://pytorch.org/docs/stable/generated/torch.randn.html
            #Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 
            
            '''
            torch.randn(2, 3)
            OUTPUT: tensor([[ 1.5954,  2.8929, -1.0923],
                            [ 1.1719, -0.4709, -0.1996]])
            '''

        else:
            self.xavier_value = torch.sqrt(torch.FloatTensor([1.0 / (self.visible_units + self.hidden_units)])).to(DEVICE)
            self.W = -self.xavier_value + torch.rand(self.visible_units, self.hidden_units).to(DEVICE) * (2 * self.xavier_value)
        self.h_bias = torch.zeros(self.hidden_units).to(DEVICE) #hidden layer bias
        self.v_bias = torch.zeros(self.visible_units).to(DEVICE) #visible layer bias
        self.h_linear_classifier = LinearClassifier(input_dim=self.hidden_units, output_dim=10).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

    def to_hidden(self ,X):

       #molto facile come funzione

        '''
        Converts the data in visible layer to hidden layer
        also does sampling
        X here is the visible probabilities
        :param X: torch tensor shape = (n_samples , n_features)
        :return -  X_prob - new hidden layer (probabilities)
                    sample_X_prob - Gibbs sampling of hidden (1 or 0) based
                                on the value
        '''
        X.to(self.Device)
        self.W.to(self.Device)

        X_prob = torch.matmul(X,self.W)
        X_prob = torch.add(X_prob, self.h_bias)#W.x + c
        X_prob  = torch.sigmoid(X_prob)

        sample_X_prob = self.sampling(X_prob)

        return X_prob,sample_X_prob

    def to_visible(self,X):
        '''
        reconstructs data from hidden layer
        also does sampling
        X here is the probabilities in the hidden layer
        :returns - X_prob - the new reconstructed layers(probabilities)
                    sample_X_prob - sample of new layer(Gibbs Sampling)

        '''
        X.to(self.Device)

        # computing hidden activations and then converting into probabilities
        X_prob = torch.matmul(X ,self.W.transpose( 0 , 1) )
        X_prob = torch.add(X_prob , self.v_bias)
        X_prob = torch.sigmoid(X_prob)

        sample_X_prob = self.sampling(X_prob)

        return X_prob,sample_X_prob

    def sampling(self,prob):
        '''
        Bernoulli sampling done based on probabilities s
        '''
        s = torch.distributions.Bernoulli(prob).sample()
        return s

    def reconstruction_error(self , data):
        '''
        Computes the reconstruction error for the data
        handled by pytorch by loss functions
        '''
        return self.contrastive_divergence(data, False)

    def reconstruct(self , X,  n_gibbs, gather_h_data=False, etichetta=11, is_train_set=True):
        '''
        This will reconstruct the sample with k steps of gibbs Sampling

        '''
        v = X
        v = v.to(self.Device)
        for i in range(n_gibbs):
            prob_h_,h = self.to_hidden(v) # DOMANDA: la ricorstruzione hidden va fatta a partire da v o hidden v?

            if i==n_gibbs-1:

                if gather_h_data:
                    if is_train_set:
                        self.h_train_labels.append(etichetta)
                        try:
                            self.h_train_dataset = torch.cat((self.h_train_dataset, prob_h_), dim=0)
                        except:
                            self.h_train_dataset = prob_h_

                    else:
                        self.h_test_labels.append(etichetta)
                        try:
                            self.h_test_dataset = torch.cat((self.h_test_dataset, prob_h_), dim=0)
                        except:
                            self.h_test_dataset = prob_h_

            prob_v_,v = self.to_visible(prob_h_)

        return prob_v_,v

    def h_from_label(self, label=5, multiplier = 1):

        #si vede qualcosa, ma non funziona super bene
        
        lbl_vec = torch.zeros(10).to(self.Device)
        lbl_vec[label]=1*multiplier
        lbl_vec = lbl_vec.cpu().numpy()

        biased_h =np.dot(self.W_inv,lbl_vec)

        v, sample_v = self.to_visible(torch.from_numpy(biased_h).to(self.Device))

        reconstructed_img = sample_v.view((28,28)).cpu()

        plt.imshow(reconstructed_img , cmap = 'gray')

        return biased_h

    def reconstruct_from_h(self,lbl,nr_steps = 50, nr_print=5):

        figure, axis = plt.subplots(1, nr_print+1, figsize=(3*(nr_print+1),3))


        print_idx = list(range(round(nr_steps/nr_print)-1,nr_steps,round(nr_steps/nr_print)))

        biased_h = self.h_from_label(label=lbl, multiplier = 1)

        prob_v_, sample_v = self.to_visible(torch.from_numpy(biased_h).to(self.Device))

        reconstructed_img = sample_v.view((28,28)).cpu()

        axis[0].imshow(reconstructed_img, cmap = 'gray')
        axis[0].set_title(str(lbl)+" after {} reconstructions".format(1))

        axis[0].set_xticklabels([])
        axis[0].set_yticklabels([])
        axis[0].set_aspect('equal')



        counter = 1
        
        for i in range(nr_steps):

            prob_h_,h = self.to_hidden(sample_v)

            prob_v_,sample_v = self.to_visible(prob_h_)

            if (i in print_idx):
                reconstructed_img = sample_v.view((28,28)).cpu()

                axis[counter].imshow(reconstructed_img, cmap = 'gray')
                axis[counter].set_title(str(lbl)+" after {} reconstructions".format(i+1))

                axis[counter].set_xticklabels([])
                axis[counter].set_yticklabels([])
                axis[counter].set_aspect('equal')
                counter +=1
        
        return figure, axis


    def reset_h_tran_test_set(self,train=False, test=True):
        if train:
            self.h_train_labels = []
            delattr(self, 'h_train_dataset' )
        

        if test:
            self.h_test_labels = []
            delattr(self, 'h_test_dataset')
            delattr(self, 'nr_gibbs_test')



    def create_h_tran_test_set(self, data, labels, nr_train_el=48000, nr_test_el=12000, nr_gibbs=1):
        if nr_train_el>0:
            for nr in range(nr_train_el):
                idx = random.randint(0,len(data)-1)
                img = data[idx]
                lbl = labels[idx]
                reconstructed_img = img.view(1,-1).type(torch.FloatTensor)

                lbl = lbl.numpy()
                lbl = int(lbl)

                _,reconstructed_img= self.reconstruct(reconstructed_img, nr_gibbs, True, lbl, is_train_set=True)

        if nr_test_el>0:
            for nr in range(nr_test_el):
                idx = random.randint(0,len(data)-1)
                img = data[idx]
                lbl = labels[idx]
                reconstructed_img = img.view(1,-1).type(torch.FloatTensor)

                lbl = lbl.numpy()
                lbl = int(lbl)

                _,reconstructed_img= self.reconstruct(reconstructed_img, nr_gibbs, True, lbl, is_train_set=False)
                self.nr_gibbs_test = nr_gibbs


    def train_h_Linear_classifier(self, nr_cat=10):

        #From Modeling language and cognition with deep unsupervised learning: a tutorial overview (Zorzi et al, 2013)

        P = self.h_train_dataset.T
        P_plus = torch.linalg.pinv(P).cpu().numpy()

        L = torch.zeros(nr_cat,len(self.h_train_dataset))
        c=0
        for lbl in self.h_train_labels:
            L[lbl,c]=1
            c=c+1
    
        L_plus = torch.linalg.pinv(L).cpu().numpy()

        W = np.dot(L,P_plus)
        W_inv = np.dot(P.cpu().numpy(),L_plus)

        with torch.no_grad():
            self.h_linear_classifier.linear.weight.copy_(torch.from_numpy(W))

        self.W_inv = W_inv
        
            
    def test_h_Linear_classifier(self):
        tensor_X_test = self.h_test_dataset.type(torch.FloatTensor).to(self.Device) # transform to torch tensors
        y_test = torch.from_numpy(np.array(self.h_test_labels))
        tensor_y_test = y_test.type(torch.LongTensor).to(self.Device)

        test_dataset = torch.utils.data.TensorDataset(tensor_X_test, tensor_y_test) # create your datset

        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=50,shuffle=True) # create your dataloader

        correct, total = 0, 0

        with torch.no_grad():

            for images, labels in test_dataloader:
                output = self.h_linear_classifier(images.view(images.shape[0], -1))

                _, predicted = torch.max(output.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (100 * correct / total)


    
    def contrastive_divergence(self, input_data ,training = True,
                                n_gibbs_sampling_steps=1,lr = 0.001):
        # positive phase

        input_data.to(self.Device)

        #qui sta l'errore
        positive_hidden_probabilities,positive_hidden_act  = self.to_hidden(input_data)

        # calculating W via positive side
        positive_associations = torch.matmul(input_data.t() , positive_hidden_act)



        # negetive phase
        hidden_activations = positive_hidden_act
        hidden_activations.to(self.Device)
        for i in range(n_gibbs_sampling_steps):
            visible_probabilities , _ = self.to_visible(hidden_activations)
            hidden_probabilities,hidden_activations = self.to_hidden(visible_probabilities)

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        # calculating W via negative side
        negative_associations = torch.matmul(negative_visible_probabilities.t() , negative_hidden_probabilities)


        # Update parameters
        if(training):

            batch_size = self.batch_size

            g = (positive_associations - negative_associations)
            grad_update = g / batch_size
            v_bias_update = torch.sum(input_data - negative_visible_probabilities,dim=0)/batch_size
            h_bias_update = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities,dim=0)/batch_size

            self.W += lr * grad_update
            self.v_bias += lr * v_bias_update
            self.h_bias += lr * h_bias_update


        # Compute reconstruction error
        error = torch.mean(torch.sum((input_data - negative_visible_probabilities)**2 , dim = 0))

        return error,torch.sum(torch.abs(grad_update))


    def forward(self,input_data):
        'data->hidden'
        return  self.to_hidden(input_data)

    def step(self,input_data,epoch,num_epochs):
        '''
            Includes the foward prop plus the gradient descent
            Use this for training
        '''
        if self.increase_to_cd_k:
            n_gibbs_sampling_steps = int(math.ceil((epoch/num_epochs) * self.k))
        else:
            n_gibbs_sampling_steps = self.k

        if self.learning_rate_decay:
            lr = self.learning_rate / epoch
        else:
            lr = self.learning_rate

        return self.contrastive_divergence(input_data , True,n_gibbs_sampling_steps,lr);


    def train(self,train_dataloader , num_epochs = 50,batch_size=16, nr_data=60000):

        self.nr_train_epochs_done = self.nr_train_epochs_done+ num_epochs

        self.batch_size = batch_size
        if(isinstance(train_dataloader ,torch.utils.data.DataLoader)):
            train_loader = train_dataloader
        else:
            train_loader = torch.utils.data.DataLoader(train_dataloader, batch_size=batch_size)


        Avg_cost = torch.FloatTensor(num_epochs , 1).to(self.Device)
        Std_cost = torch.FloatTensor(num_epochs , 1).to(self.Device)

        for epoch in range(1 , num_epochs+1):
            epoch_err = 0.0
            n_batches = int(len(train_loader))
            # print(n_batches)

            cost_ = torch.FloatTensor(n_batches , 1).to(self.Device)
            grad_ = torch.FloatTensor(n_batches , 1).to(self.Device)

            for i,(batch,_) in tqdm(enumerate(train_loader),ascii=True,
                                desc="RBM fitting", file=sys.stdout):

                batch = batch.view(len(batch) , self.visible_units)
                #print(batch.shape) #debug
                batch = batch.to(self.Device)
                '''
                old code (prima di inserire batch = batch.to(self.Device))
                if(self.use_gpu):
                    batch = batch.cuda()                
                '''

                cost_[i-1],grad_[i-1] = self.step(batch,epoch,num_epochs)


            Avg_cost[epoch-1] = torch.mean(cost_)
            Std_cost[epoch-1] = torch.std(cost_)

            print("Epoch:{} ,avg_cost = {} ,std_cost = {} ,avg_grad = {} ,std_grad = {}".format(epoch,\
                                                            torch.mean(cost_),\
                                                            torch.std(cost_),\
                                                            torch.mean(grad_),\
                                                            torch.std(grad_)))

        
        if len(self.RBM_train_loss)>0:
            self.RBM_train_loss[0] = self.RBM_train_loss[0] + Avg_cost.cpu().numpy().tolist()
            self.RBM_train_loss_std[0] = self.RBM_train_loss_std[0] + Std_cost.cpu().numpy().tolist()

        else:
            self.RBM_train_loss.append(Avg_cost.cpu().numpy().tolist())
            self.RBM_train_loss_std.append(Std_cost.cpu().numpy().tolist())

        plt.plot(self.RBM_train_loss[0], '-', lw=2)
        plt.xlabel('epoch')
        plt.ylabel('reconstruction error (MSE)')
        plt.title('RBM - training curve')
        '''
        ERROR BARS NON FUNZIONANTI - DA mettere a posto quando hai tempo
        ymin = np.array(self.RBM_train_loss[0]) - np.array(self.RBM_train_loss_std[0])/np.sqrt(nr_data) #SEM
        ymax = np.array(self.RBM_train_loss[0]) + np.array(self.RBM_train_loss_std[0])/np.sqrt(nr_data)

        x=list(range(0,len(self.RBM_train_loss)))

        plt.fill_between(x, ymax, ymin)

        '''


        plt.show() 





        return Avg_cost, Std_cost

    def save_model(self, nr_gibbs_htrain=1, nr_gibbs_htest=1):
        #lavora con drive

        try:
            h_train_size = self.h_train_dataset.shape[0]
        except:
            h_train_size = 0

        try:
            h_test_size = self.h_test_dataset.shape[0]
        except:
            h_test_size = 0


        self.filename = 'rbm_train'+ str(self.nr_train_epochs_done)+'_generated_h_train'+str(h_train_size)+'gibbs'+str(nr_gibbs_htrain)+'_generated_h_test'+str(h_test_size)+'gibbs'+str(nr_gibbs_htest)

        object = self
 

        from google.colab import drive
        drive.mount('/content/gdrive')

        save_path = "/content/gdrive/My Drive/"+self.filename

        try:
            os.mkdir(save_path)
        except:
            print("Folder already found")

        Filename = save_path +'/'+ self.filename + '.pkl'

        with open(Filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
        
        
        
        




