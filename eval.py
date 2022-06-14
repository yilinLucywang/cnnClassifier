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