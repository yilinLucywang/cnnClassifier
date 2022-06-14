import nltk
import torch
import numpy as np
from nltk.tokenize import word_tokenize
import re

labels = {'translate':0,
          'transfer':1,
          'timer':2,
          'definition':3,
          'meaning_of_life':4,
          'insurance_change':5,
          'find_phone':6,
          'travel_alert':7,
          'pto_request':8,
          'improve_credit_score':9,
          'fun_fact':10, 
          'change_language':11,
          'payday':12,
          'replacement_card_duration':13,
          'time':14,
          'application_status':15,
          'flight_status':16,
          'flip_coin':17,
          'change_user_name':18, 
          'where_are_you_from':19,
          'shopping_list_update':20,
          'what_can_i_ask_you':21,
          'maybe':22,
          'oil_change_how':23,
          'restaurant_reservation':24,
          'balance':25,
          'confirm_reservation':26,
          'freeze_account':27,
          'rollover_401k':28,
          'who_made_you':29,
          'distance':30,
          'user_name':31,
          'timezone':32,
          'next_song':33,
          'transactions':34,
          'restaurant_suggestion':35,
          'rewards_balance':36,
          'pay_bill':37,
          'spending_history':38,
          'pto_request_status':39,
          'credit_score':40,
          'new_card':41,
          'lost_luggage':42,
          'repeat':43,
          'mpg':44,
          'oil_change_when':45,
          'yes':46,
          'travel_suggestion':47,
          'insurance':48,
          'todo_list_update':49,
          'reminder':50,
          'change_speed':51,
          'tire_pressure':52,
          'no':53,
          'apr':54,
          'nutrition_info':55,
          'calendar':56,
          'uber':57,
          'calculator':58,
          'date':59,
          'carry_on':60,
          'pto_used':61,
          'schedule_maintenance':62,
          'travel_notification':63,
          'sync_device':64,
          'thank_you':65,
          'roll_dice':66,
          'food_last':67,
          'cook_time':68,
          'reminder_update':69,
          'report_lost_card':70,
          'ingredient_substitution':71,
          'make_call':72,
          'alarm':73,
          'todo_list':74,
          'change_accent':75,
          'w2':76,
          'bill_due':77,
          'calories':78,
          'damaged_card':79,
          'restaurant_reviews':80,
          'routing':81,
          'do_you_have_pets':82,
          'schedule_meeting':83,
          'gas_type':84,
          'plug_type':85,
          'tire_change':86,
          'exchange_rate':87,
          'next_holiday':88,
          'change_volume':89,
          'who_do_you_work_for':90,
          'credit_limit':91,
          'how_busy':92,
          'accept_reservations':93,
          'order_status':94,
          'pin_change':95,
          'goodbye':96,
          'account_blocked':97,
          'what_song':98,
          'international_fees':99,
          'last_maintenance':100,
          'meeting_schedule':101,
          'ingredients_list':102,
          'report_fraud':103,
          'measurement_conversion':104,
          'smart_home':105,
          'book_hotel':106,
          'current_location':107,
          'weather':108,
          'taxes':109,
          'min_payment':110,
          'whisper_mode':111,
          'cancel':112,
          'international_visa':113,
          'vaccines':114,
          'pto_balance':115,
          'directions':116,
          'spelling':117,
          'greeting':118,
          'reset_settings':119,
          'what_is_your_name':120,
          'direct_deposit':121,
          'interest_rate':122,
          'credit_limit_change':123,
          'what_are_your_hobbies':124,
          'book_flight':125,
          'shopping_list':126,
          'text':128,
          'bill_balance':129,
          'share_location':130,
          'redeem_rewards':131,
          'play_music':132,
          'calendar_update':133,
          'are_you_a_bot':134,
          'gas':135,
          'expiration_date':136,
          'update_playlist':137,
          'cancel_reservation':138,
          'tell_joke':139,
          'change_ai_name':140,
          'how_old_are_you':141,
          'car_rental':142,
          'jump_start':143,
          'meal_suggestion':144,
          'recipe':145,
          'income':146,
          'order':147,
          'traffic':148,
          'order_checks':149,
          'card_declined':150
          }


class Preprocessing:
    def __init__(self, num_words, seq_len):
      #check what this nom_words means
      #TODO: need to change to dynamic padding
      self.num_words = num_words
      self.seq_len = seq_len
      self.vocabulary = None
      self.x_tokenized = None
      self.x_padded = None
      
      self.t_train = None
      self.t_test = None
      self.l_train = None
      self.l_test = None


      self.t_train_tkn = None
      self.t_test_tkn = None
      self.l_train_tkn = None
      self.l_test_tkn = None
      
      self.load_data()



    def load_data(self):
      train_path = "/Users/wangyilin/Desktop/CNNclassifier/data/ind_try"
      test_path = "/Users/wangyilin/Desktop/CNNclassifier/data/ind_try"
      #load training data
      f = open(train_path, "r")
      content = f.read()

      content = re.sub(r'[^\w\s]', '', content)

      phraseList = content.split("\n")
      #self.labels = np.empty(len(phraseList), dtype=np.float64)
      curTexts = []
      curLabels = []
      for phrase in phraseList:
          p = phrase.split("\t")
          #print(p)
          #np.append(self.labels,p[0])
          if len(p) == 2: 
              curTexts.append(p[1])
              curLabels.append(p[0])
          else: 
              continue
      self.l_train = [int(labels[label]) for label in curLabels]
      self.lens = [len(text) for text in curTexts]
      self.t_train = [word_tokenize(text) for text in curTexts]

      #load testing data
      f = open(test_path, "r")
      content = f.read()
      phraseList = content.split("\n")
      #self.labels = np.empty(len(phraseList), dtype=np.float64)
      curTexts = []
      curLabels = []
      for phrase in phraseList:
          p = phrase.split("\t")
          #print(p)
          #np.append(self.labels,p[0])
          if len(p) == 2: 
              curTexts.append(p[1])
              curLabels.append(p[0])
          else: 
              continue
      self.l_test = [int(labels[label]) for label in curLabels]
      self.lens = [len(text) for text in curTexts]
      self.t_test = [word_tokenize(text) for text in curTexts]


      self.build_vocabulary()
      self.word_to_idx()
      self.padding_sentences()
      print("printing here")
      print(self.t_test)



    def build_vocabulary(self):
        # Check what should be inside the x_raw
        self.vocabulary = dict()
        fdist = nltk.FreqDist()
        #check whether the t_train work here
        for sentence in self.t_train:
            for word in sentence:
                fdist[word] += 1

        for sentence in self.t_test: 
            for word in sentence:
                fdist[word] += 1

        common_words = fdist.most_common(self.num_words)
        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = (idx+1)

    def word_to_idx(self):
      # By using the dictionary (vocabulary), it is transformed
      # each token into its index based representation  
      #self.x_tokenized = list()
      self.t_train_tkn = list()
      self.t_test_tkn = list()

      #stores t_train
      for sentence in self.t_train:
        temp_sentence = list()
        for word in sentence:
           if word in self.vocabulary.keys():
              #temp_sentence stores a list of index
              print(word)
              print(self.vocabulary[word])
              temp_sentence.append(self.vocabulary[word])
        self.t_train_tkn.append(temp_sentence)

      #stores t_test
      for sentence in self.t_test:
        temp_sentence = list()
        for word in sentence:
           if word in self.vocabulary.keys():
              temp_sentence.append(self.vocabulary[word])
        self.t_test_tkn.append(temp_sentence)

    def padding_sentences(self):
      print("seq_len is: ")
      print(self.seq_len)
      pad_idx = 0
      self.t_test = list()
      for sentence in self.t_train_tkn:
        while len(sentence) < self.seq_len:
          sentence.insert(len(sentence), pad_idx)
        self.t_test.append(sentence)
      self.t_test = np.array(self.t_test)


      pad_idx = 0
      self.t_train = list()
      for sentence in self.t_test_tkn:
        while len(sentence) < self.seq_len:
          sentence.insert(len(sentence), pad_idx)
        self.t_train.append(sentence)
      self.t_train = np.array(self.t_train)


