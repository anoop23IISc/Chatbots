import json
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import bert
# from bert import tokenization
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math
import pickle
import time
from schema import *
import sys


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



# data_path="/home/abdul/BT/SGD/DA_BT_data/train_test_scripts/"





dialogue_act_embed = ""
with open('./acts_embed.pickle', 'rb') as handle:
    dialogue_act_embed = pickle.load(handle)
dialogue_acts = list(dialogue_act_embed.keys())



from bert import bert_tokenization
# FullTokenizer = tokenization.FullTokenizer
FullTokenizer = bert_tokenization.FullTokenizer



max_length = 80  # Your choice here.

input_word_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32,
                                    name="segment_ids")

# bert_layer = hub.KerasLayer("D:\\AI\\bertmodel\\bertmodel", trainable=False)



bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1",trainable=False)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])





model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def truncate(tokens_a,tokens_b):
    is_too_long = False
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length-3:
          break
        is_too_long = True
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()
    return is_too_long

def get_inputs(utterance1,utterance2):
    tokens_a = tokenizer.tokenize(utterance1)
    tokens_b = tokenizer.tokenize(utterance2)
    truncate(tokens_a,tokens_b)
    
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)


    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    assert len(input_type_ids) == max_length
    return input_ids,input_mask,input_type_ids


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
         
"""         
def get_dialogue_dataset(file_path,dialogue_dataset):
    fp = open(file_path, 'r')
    data = json.load(fp)
    #print ("file_path, data")
    #print (file_path, data)
    for dialogue in data:
        turns = dialogue["turns"]
        
        prev_system_utterance = ""
        prev_system_actions = set()
        begin_turn = len(turns)-4 if len(turns) >=4 else 0
        for num in range(begin_turn,len(turns),2):
            user_turn = turns[num]
            system_turn = turns[num+1]
            datapoint = {}
            if num == begin_turn and len(turns) >=4:
                prev_system_utterance = system_turn["utterance"]
                datapoint["system_actions"] = set()
                for action in system_turn["frames"][0]["actions"]:
                    action_name = action["act"]+":"+action["slot"] if action["act"] else {}
                    if action_name:
                        datapoint["system_actions"].add(action_name)
                prev_system_actions = datapoint["system_actions"]
                continue
            
            datapoint["prev_system_actions"] = prev_system_actions
            datapoint["previous_system_utterance"] = prev_system_utterance
            
            datapoint["user_utterance"] = user_turn["utterance"]
            datapoint["system_actions"] = set()
            for action in system_turn["frames"][0]["actions"]:
                action_name = action["act"]+":"+action["slot"] if action["act"] else {}
                if action_name:
                    datapoint["system_actions"].add(action_name)

            datapoint["intents"] = set()
            datapoint["slots"] = set()
            datapoint["requestable_slots"]=set()
            for service in user_turn["frames"]:
                service_name = service["service"]
                for non_cat_slot in service["slots"]:
                    slot_name = non_cat_slot["slot"]
                    datapoint["slots"].add(service_name+":"+slot_name)
                for cat_slot_name in service["state"]["slot_values"]:
                    datapoint["slots"].add(service_name+":"+cat_slot_name)
                for req_slot_name in service["state"]["requested_slots"]:
                    datapoint["requestable_slots"].add(service_name+":"+req_slot_name)
                intent_name = service["state"]["active_intent"]
                
                
                if intent_name!="NONE" and intent_name!="":
                    datapoint["intents"].add(service_name+":"+intent_name)     
            dialogue_dataset.append(datapoint)
            
            prev_system_utterance = system_turn["utterance"]
            prev_system_actions = datapoint["system_actions"]

"""




def get_dialogue_dataset(file_path,dialogue_dataset):
    fp = open(file_path, 'r')
    data = json.load(fp)
    #print ("file_path, data")
    #print (file_path, data)
    for dialogue in data:
        turns = dialogue["turns"]
        
        prev_system_utterance = ""
        prev_system_actions = set()
        
        prev_user_intent = set()
        
        
        belief_state = set()
        
        
        intent_window = list()
        
        
        
        for num in range(0,len(turns),2):
            user_turn = turns[num]
            system_turn = turns[num+1]
            
            
            
            datapoint = {}
            
            
            datapoint["belief_state"] = belief_state
            
            
            datapoint["prev_user_intent"]=set()
            datapoint["prev_system_actions"] = prev_system_actions
            datapoint["previous_system_utterance"] = prev_system_utterance
            datapoint["intent_window"] = intent_window
            
            
            
            datapoint["user_utterance"] = user_turn["utterance"]
            datapoint["system_actions"] = set()
            for action in system_turn["frames"][0]["actions"]:
                action_name = action["act"]+":"+action["slot"] if action["act"] else {}
                if action_name:
                    datapoint["system_actions"].add(action_name)

                    
            datapoint["intents"] = set()
            datapoint["slots"] = set()
            datapoint["requestable_slots"]=set()
            
            
            for service in user_turn["frames"]:
                service_name = service["service"]
                for non_cat_slot in service["slots"]:
                    slot_name = non_cat_slot["slot"]
                    datapoint["slots"].add(service_name+":"+slot_name)
                
                for cat_slot_name in service["state"]["slot_values"]:
                    datapoint["slots"].add(service_name+":"+cat_slot_name)
                
                
                
                for req_slot_name in service["state"]["requested_slots"]:
                    datapoint["requestable_slots"].add(service_name+":"+req_slot_name)
          
                
                
                intent_name = service["state"]["active_intent"]
                if intent_name!="NONE" and intent_name!="":
                    datapoint["intents"].add(service_name+":"+intent_name)  
                    
#                 if prev_user_intent !="" and prev_user_intent !="NONE":
#                     datapoint["prev_user_intent"].add(service_name+":"+prev_user_intent)  
                    
            
            
            datapoint["prev_user_intent"] = prev_user_intent
            dialogue_dataset.append(datapoint)    
        
        
            for intent in datapoint["intents"]:
                intent_window.append(intent)
                if len(intent_window)>3:
                    intent_window.pop(0)
        
            prev_system_utterance = system_turn["utterance"]
            prev_system_actions = datapoint["system_actions"]
            prev_user_intent = datapoint["intents"]
            belief_state = belief_state.union(datapoint["slots"])


            
            
            

def create_data(data_type,pickle_name):
    
    
    print("Creating "+data_type+" data")
    
    
    data_type_path = root_path+data_type+"/"
    
    print(data_type_path+'schema.json')
    
    
    
    onlyfiles = [f for f in listdir(data_type_path) if isfile(join(data_type_path, f))]
    dialogue_dataset=[]
    for file in onlyfiles:
        print(file, end =", ")
        if file in ['schema.json','desktop.ini','schema.json.bak']:
            continue
        else:
            file_path = data_type_path+file
            get_dialogue_dataset(file_path,dialogue_dataset)
    print("\n")
    print(len(dialogue_dataset))
    print("Printing random datapoint")
    pretty(random.choice(dialogue_dataset))

    
    input_ids_list=[]
    input_masks_list = []
    input_segments_list = []
    
    
    
    print("Creating Bert Input data")
    start = time.time()
    for i in dialogue_dataset:
        
#         break
        user_utterance = i["user_utterance"]
        
        system_utterance = i["previous_system_utterance"]
#         system_utterance = ""
        
        
        input_ids,input_mask,input_type_ids = get_inputs(user_utterance,system_utterance)
        input_ids_list.append(input_ids)
        input_masks_list.append(input_mask)
        input_segments_list.append(input_type_ids)    
    print('Time taken {} secs\n'.format(time.time() - start))


    input_ids_tensor=tf.convert_to_tensor(input_ids_list)
    input_masks_tensor=tf.convert_to_tensor(input_masks_list)
    input_segments_tensor=tf.convert_to_tensor(input_segments_list)
    sentence_embed_list = []
    eval_batch_size = 10000
    
    
    
    
    
    
    
    print("Creating Bert Embeddings")
    start = time.time()
    for batch_num in range(int(np.ceil(len(input_ids_list)/eval_batch_size))):
            print(batch_num)
#             break
            start_num = batch_num*eval_batch_size
            end_num = start_num + eval_batch_size
            if end_num>len(input_ids_list):
                end_num = len(input_ids_list)
    #         with tf.device('/device:GPU:3'): #much more time
            if True: #495.01423716545105 secs
                pool_embs, all_embs = model.predict([input_ids_tensor[start_num:end_num],input_masks_tensor[start_num:end_num],input_segments_tensor[start_num:end_num]])
            print(pool_embs.shape)
            pool_embs = pool_embs.tolist()
            sentence_embed_list+=pool_embs
            print(len(sentence_embed_list))
            print('Time taken {} secs\n'.format(time.time() - start))
    
    
    print('Time taken {} secs\n'.format(time.time() - start))
    assert len(sentence_embed_list)==len(dialogue_dataset)



    schema_file_path = data_type_path+'schema.json'
    schema = Schema(schema_file_path)
    schema_embed_filename = embed_path+data_type+'_pretrained_schema_embedding.npy'
    schema_embeddings = np.load(schema_embed_filename,allow_pickle=True) 


    print("Creating the formatted data")
    start = time.time()
    formatted_data = []
    for num,val in enumerate(dialogue_dataset):
        data_point = {}
        data_point["system_actions"] = val["system_actions"]
        data_point["utterance_emb"] = np.array(sentence_embed_list[num]).reshape(1,768)

        
        
        intent_emb = np.zeros((1,768))
        for intent in val["intents"]:
            service_name = intent.split(":")[0]
            intent_name = intent.split(":")[1]
            service_id = schema.get_service_id(service_name)
            intent_id = schema._service_schemas[service_name]._intents.index(intent_name)
            embed_val = schema_embeddings[service_id]["intent_emb"][intent_id].reshape((1,768))
            intent_emb += embed_val
        if len(val["intents"])!=0:
            intent_emb = intent_emb/len(val["intents"])
        data_point["intent_emb"] = intent_emb
        
        
        
        prev_intent_emb = np.zeros((1,768))
        for intent in val["prev_user_intent"]:
            service_name = intent.split(":")[0]
            intent_name = intent.split(":")[1]
            service_id = schema.get_service_id(service_name)
            intent_id = schema._service_schemas[service_name]._intents.index(intent_name)
            embed_val = schema_embeddings[service_id]["intent_emb"][intent_id].reshape((1,768))
            prev_intent_emb += embed_val
        if len(val["prev_user_intent"])!=0:
            prev_intent_emb = prev_intent_emb/len(val["prev_user_intent"])
        data_point["prev_intent_emb"] = prev_intent_emb
          
            
            
            
            
            
            
        intent_window_emb = np.zeros((1,768))
        for intent in val["intent_window"]:
            service_name = intent.split(":")[0]
            intent_name = intent.split(":")[1]
            service_id = schema.get_service_id(service_name)
            intent_id = schema._service_schemas[service_name]._intents.index(intent_name)
            embed_val = schema_embeddings[service_id]["intent_emb"][intent_id].reshape((1,768))
            intent_window_emb += embed_val
        if len(val["intent_window"])!=0:
            intent_window_emb = intent_window_emb/len(val["intent_window"])
        data_point["intent_window_emb"] = intent_window_emb
            
            
          
        
        req_emb = np.zeros((1,768))
        for slots in val['requestable_slots']:
            service_name = slots.split(":")[0]
            slot_name = slots.split(":")[1]
            service_id = schema.get_service_id(service_name)
            slot_id = schema._service_schemas[service_name]._slots.index(slot_name)
            embed_val = schema_embeddings[service_id]["req_slot_emb"][slot_id].reshape((1,768))
            req_emb += embed_val
        if len(val["requestable_slots"])!=0:
            req_emb = req_emb/len(val["requestable_slots"])
        data_point["req_emb"] = req_emb
               

            

        slots_emb = np.zeros((1,768))
        for slots in val['slots']:
            service_name = slots.split(":")[0]
            slot_name = slots.split(":")[1]
            service_id = schema.get_service_id(service_name)
            slot_id = schema._service_schemas[service_name]._slots.index(slot_name)
            embed_val = schema_embeddings[service_id]["req_slot_emb"][slot_id].reshape((1,768))
            slots_emb += embed_val
        if len(val["slots"])!=0:
            slots_emb = slots_emb/len(val["slots"])
        data_point["slots_emb"] = slots_emb
        
        
        
        
        
        belief_state_emb = np.zeros((1,768))
        for slots in val['belief_state']:
            service_name = slots.split(":")[0]
            slot_name = slots.split(":")[1]
            service_id = schema.get_service_id(service_name)
            slot_id = schema._service_schemas[service_name]._slots.index(slot_name)
            embed_val = schema_embeddings[service_id]["req_slot_emb"][slot_id].reshape((1,768))
            belief_state_emb += embed_val
        if len(val["belief_state"])!=0:
            belief_state_emb = belief_state_emb/len(val["belief_state"])
        data_point["belief_state"] = belief_state_emb
        
        
        

        prev_action_emb = np.zeros((1,768))
        for prev_act in val["prev_system_actions"]:
            prev_action_emb += dialogue_act_embed[prev_act]
        if len(val["prev_system_actions"])!=0:
            prev_action_emb = prev_action_emb/len(val["prev_system_actions"])
        data_point["prev_action_emb"] = prev_action_emb
        formatted_data.append(data_point)

        
        
        
    print('Time taken {} secs\n'.format(time.time() - start))
    assert len(formatted_data)==len(dialogue_dataset)
    
    print("Saving as pickle file")
    with open(pickle_name, 'wb') as handle:
        pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
            
# root_path = data_path + "data/" #sys.argv[1]
# embed_path = data_path + "embed_data/" #sys.argv[2]
# create_data("train","./data/training_data_bs_i3.pickle")
# create_data("dev","dev_data.pickle")


root_path = '../dstc8-schema-guided-dialogue/'
embed_path = '../output_schema_embedding_dir/'
create_data("train","training_data.pickle")
create_data("dev","dev_data.pickle")  

