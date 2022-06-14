from torch.utils.data import Dataset, DataLoader
from loadData import *
from model import *
from torch import optim
import re
import torch.nn.functional as F
class DatasetMaper(Dataset):
   def __init__(self, x, y):
      self.x = x
      self.y = y
      
   def __len__(self):
      return len(self.x)
      
   def __getitem__(self, idx):
      return self.x[idx], self.y[idx]


class Run:
   '''Training, evaluation and metrics calculation'''

   @staticmethod
   def train(model, data, params):
      
      # Initialize dataset maper
      train = DatasetMaper(data.t_train, data.l_train)
      test = DatasetMaper(data.t_test, data.l_test)
      
      # Initialize loaders
      loader_train = DataLoader(train, batch_size=params.batch_size)
      loader_test = DataLoader(test, batch_size=params.batch_size)
      
      # Define optimizer
      optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
      
      # Starts training phase
      for epoch in range(params.epochs):
         # Set model in training model
         model.train()
         predictions = []
         # Starts batch training
         for x_batch, y_batch in loader_train:
         
            y_batch = y_batch.type(torch.FloatTensor)
            
            # Feed the model
            y_pred = model(x_batch)
            
            # Loss calculation
            loss = F.binary_cross_entropy(y_pred, y_batch)
            
            # Clean gradientes
            optimizer.zero_grad()
            
            # Gradients calculation
            loss.backward()
            
            # Gradients update
            optimizer.step()
            
            # Save predictions
            predictions += list(y_pred.detach().numpy())
         
         # Evaluation phase
         test_predictions = Run.evaluation(model, loader_test)
         
         # Metrics calculation
         train_accuary = Run.calculate_accuray(data.l_train, predictions)
         test_accuracy = Run.calculate_accuray(data.l_test, test_predictions)
         print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))
         
   @staticmethod
   def evaluation(model, loader_test):
      
      # Set the model in evaluation mode
      model.eval()
      predictions = []
      
      # Starst evaluation phase
      with torch.no_grad():
         for x_batch, y_batch in loader_test:
            y_pred = model(x_batch)
            predictions += list(y_pred.detach().numpy())
      return predictions
      
   @staticmethod
   def calculate_accuray(grand_truth, predictions):
      # Metrics calculation
      true_positives = 0
      true_negatives = 0
      for true, pred in zip(grand_truth, predictions):
         if (pred >= 0.5) and (true == 1):
            true_positives += 1
         elif (pred < 0.5) and (true == 0):
            true_negatives += 1
         else:
            pass
      # Return accuracy
      return (true_positives+true_negatives) / len(grand_truth)

class Params(): 
   def __init__(self, seq_len, num_words, embedding_size, batch_size):
      self.seq_len = seq_len
      self.num_words = num_words
      self.embedding_size = embedding_size
      self.batch_size = batch_size
      self.out_size =32
      #should be 1d, so one number would suffice
      self.stride = 2
      self.learning_rate = 0.001
      self.epochs = 5

def train(model, data, params):
   print("inside train")
   print(data.t_train)
   # Initialize dataset maper
   train = DatasetMaper(data.t_train, data.l_train)
   test = DatasetMaper(data.t_test, data.l_test)
   
   # Initialize loaders
   loader_train = DataLoader(train, batch_size=params.batch_size)
   loader_test = DataLoader(test, batch_size=params.batch_size)
   
   # Define optimizer
   optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
   
   # Starts training phase
   for epoch in range(params.epochs):
      # Set model in training model
      model.train()
      predictions = []
      # Starts batch training
      for x_batch, y_batch in loader_train:
      
         y_batch = y_batch.type(torch.FloatTensor)
         
         # Feed the model
         y_pred = model(x_batch)
         
         # Loss calculation
         loss = F.binary_cross_entropy(y_pred, y_batch)
         
         # Clean gradientes
         optimizer.zero_grad()
         
         # Gradients calculation
         loss.backward()
         
         # Gradients update
         optimizer.step()
         
         # Save predictions
         predictions += list(y_pred.detach().numpy())
      
      # Evaluation phase
      test_predictions = Run.evaluation(model, loader_test)
      
      # Metrics calculation
      train_accuary = Run.calculate_accuray(data.l_train, predictions)
      test_accuracy = Run.calculate_accuray(data.l_test, test_predictions)

      print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))


def calculate_accuray(grand_truth, predictions):
   true_positives = 0
   true_negatives = 0
   
   # Gets frequency  of true positives and true negatives
   # The threshold is 0.5
   for true, pred in zip(grand_truth, predictions):
      if (pred >= 0.5) and (true == 1):
         true_positives += 1
      elif (pred < 0.5) and (true == 0):
         true_negatives += 1
      else:
         pass
   # Return accuracy
   return (true_positives+true_negatives) / len(grand_truth)

def get_num_words(path_list):
   word_set = set()
   for path in path_list: 
      with open(path) as fp: 
         line = fp.readline()
         line = re.sub(r'[^\w\s]', '', line)
         while line:
            print(line) 
            line_list = line.split("\t")
            if len(line_list) == 2: 
               word_list = (line_list[1].split(" "))
               for word in word_list: 
                  l_word = word.lower()
                  word_set.add(l_word)
            line = fp.readline()
            line = re.sub(r'[^\w\s]', '', line)
   return len(word_set)

def get_seq_len(path_list): 
   max_len = 0
   for path in path_list: 
      with open(path) as fp: 
         line = fp.readline()
         line = re.sub(r'[^\w\s]', '', line)
         while line: 
            line_list = line.split("\t")
            if len(line_list) == 2: 
               cur_len = len(line_list[1].split(" "))
               print("cur_len: ")
               print(line_list[1].split(" "))
               if(cur_len > max_len): 
                  max_len = cur_len
            line = fp.readline()
            line = re.sub(r'[^\w\s]', '', line)
   return max_len

def main(): 
   #TODO: count number of distinct words
   path_list = ['/Users/wangyilin/Desktop/CNNclassifier/data/ind_try', '/Users/wangyilin/Desktop/CNNclassifier/data/ind_try']
   num_words = get_num_words(path_list)
   print(num_words)
   #collect info for this params
   #seq_len is the padding size
   seq_len = get_seq_len(path_list)
   print("calculated seq_len: ")
   print(seq_len)
   #TODO: check whether 50 is enough
   embedding_size = 64
   batch_size = 12
   params = Params(seq_len, num_words, embedding_size, batch_size)
   print("start loading")
   data = Preprocessing(num_words, seq_len)
   model = TextClassifier(params)
   train(model, data, params)
main()