

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="1"



import random
import numpy as np
import tensorflow as tf
import pickle
import time
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




tf.random.set_seed(7)


# data_path="/home/abdul/BT/SGD/DA_BT_data/train_test_scripts/"
# train_pickle = data_path+"data/training_data_bs_i3.pickle"
# print("Train file:",train_pickle)
# dev_pickle=train_pickle

acts_embed_pickle = "./acts_embed.pickle"
dialogue_act_embed = []
with open(acts_embed_pickle, 'rb') as handle:
    dialogue_act_embed = pickle.load(handle)
dialogue_acts = list(dialogue_act_embed.keys())  
tf.keras.backend.set_floatx('float64')




    
class DA_Model(tf.keras.Model):
    def __init__(self):
        super(DA_Model, self).__init__()

        self.intent_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        self.slots_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        self.previous_intent_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        
        
        self.belief_state_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        #self.previous_action_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        
        
        
        
        self.requestable_slots_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        self.utterance_dense = tf.keras.layers.Dense(1024,input_shape=(768,))

        
        
        self.iw_dense = tf.keras.layers.Dense(1024,input_shape=(768,))
        
        
        
        self.concat_dense = tf.keras.layers.Dense(1024,input_shape=(768+(1024*7),)) ##
        
        
        
        self.dropout1 = tf.keras.layers.Dropout(0.0)
        self.dropout2 = tf.keras.layers.Dropout(0.0)
        
        
        do_rate1 = 0.0                                        
        self.dense_1 = tf.keras.layers.Dense(512,input_shape=(1024,))                                  
        self.dropout_3 = tf.keras.layers.Dropout(do_rate1)
        self.dense_2 = tf.keras.layers.Dense(256,input_shape=(512,))                                  
        self.dropout_4 = tf.keras.layers.Dropout(do_rate1)
        self.dense_3 = tf.keras.layers.Dense(128,input_shape=(256,))                                  
        self.dropout_5 = tf.keras.layers.Dropout(do_rate1)
        self.dense_4 = tf.keras.layers.Dense(64,input_shape=(128,))                                  
        self.dropout_6 = tf.keras.layers.Dropout(do_rate1)
        
        
        self.action_dense = tf.keras.layers.Dense(2,input_shape=(768,))

    def call(self, intent_vector,slots_vector,previous_intent_vector,action_vector,utterance_vector,prev_act_vector,
             belief_state_vector,iw_vector,requestable_slots_vector,training):
        
        
        
        intent_output = self.intent_dense(intent_vector)
        slots_output = self.slots_dense(slots_vector)
        
        requestable_slots_output =self.requestable_slots_dense(requestable_slots_vector)
        
        
        
        previous_intent_output =self.previous_intent_dense(previous_intent_vector)
        utterance_dense_ouput = self.utterance_dense(utterance_vector)
        
        
        belief_state_ouput = self.belief_state_dense(belief_state_vector)
        #previous_action_ouput = self.previous_action_dense(prev_act_vector)
        
        iw_output = self.iw_dense(iw_vector)
        
        
        
        temp_out = tf.concat([intent_output,slots_output,previous_intent_output,utterance_dense_ouput,belief_state_ouput,iw_output,requestable_slots_output], axis=1)##
        
        
        
        hidden_rep = tf.nn.relu(temp_out)
        hidden_rep = self.dropout1(hidden_rep, training=training)
        
        concatenated = tf.concat([action_vector, hidden_rep], axis=1)
        concat_output = tf.nn.relu(self.concat_dense(concatenated))
        concat_output = self.dropout2(concat_output, training=training)
        
        concat_output =  tf.nn.relu(self.dropout_3(self.dense_1(concat_output), training=training))
        concat_output =  tf.nn.relu(self.dropout_4(self.dense_2(concat_output), training=training))
        concat_output =  tf.nn.relu(self.dropout_5(self.dense_3(concat_output), training=training))
        concat_output =  tf.nn.relu(self.dropout_6(self.dense_4(concat_output), training=training))
        
        output = self.action_dense(concat_output) #for sparse categorical
    
        return output

    

def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)


@tf.function #reduces epoch time by 100 seconds
def train_step(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch):
  with tf.GradientTape() as tape:
    
    predictions = Model(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch,True)
    
    
    
    loss = loss_function(label_batch, predictions)

  gradients = tape.gradient(loss, Model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, Model.trainable_variables))
  train_loss(loss)
  train_accuracy(label_batch, predictions)
    
    
    
    
    
    
    
    
    
def get_batch_ratio(batch_data,frequent,not_frequent):    
    utterance_batch = ""
    intent_batch = ""
    slots_batch = ""
    req_batch = ""
    action_batch = ""
    label_batch = ""
    prev_act_batch = ""
    
    
    belief_state_batch = ""
    iw_batch = ""
    
    
    
    num_neg = 1
#     num_neg_freq = 2
#     num_neg_not_freq = 1
    
    
    for data in batch_data:
        utterance_emb = data["utterance_emb"]
        intent_emb = data["intent_emb"]
        slots_emb = data["slots_emb"]
        
        req_emb = data["req_emb"]
        
        
        prev_act_emb = data["prev_action_emb"]
        prev_intent_emb = data["prev_intent_emb"]
        
        
        belief_state_emb = data["belief_state"]
        iw_emb =  data["intent_window_emb"]
        
        
        positive_action = random.sample(data["system_actions"],1)
        positive_emb = dialogue_act_embed[positive_action[0]]
        label = np.array([1]).reshape(1,1)
        
        if utterance_batch =="":
            utterance_batch = utterance_emb
            intent_batch = intent_emb
            slots_batch = slots_emb
            
            req_batch = req_emb
            
            prev_act_batch = prev_act_emb
            action_batch = positive_emb
            prev_intent_batch=prev_intent_emb
            label_batch = label 
            
            
            belief_state_batch = belief_state_emb
            iw_batch = iw_emb
            
            
            
        else:
            utterance_batch = np.vstack((utterance_batch,utterance_emb))
            intent_batch = np.vstack((intent_batch,intent_emb))
            slots_batch = np.vstack((slots_batch,slots_emb))
            
            req_batch = np.vstack((req_batch,req_emb))
            
            prev_act_batch = np.vstack((prev_act_batch,prev_act_emb))
            prev_intent_batch = np.vstack((prev_intent_batch,prev_intent_emb))
            action_batch = np.vstack((action_batch,positive_emb))
            label_batch = np.vstack((label_batch,label))
            
            
            belief_state_batch = np.vstack((belief_state_batch,belief_state_emb))
            iw_batch = np.vstack((iw_batch,iw_emb))
            
        
        
        negative_actions = []
        if random.uniform(0, 1)>0.2:
            negative_actions = random.sample(list(set(frequent).difference(set(data["system_actions"]))) , num_neg)
        else:
            negative_actions = random.sample(list(set(not_frequent).difference(set(data["system_actions"]))) ,num_neg)

        
        
        
        for action  in negative_actions:
            action_emb = dialogue_act_embed[action]
            label = np.array([0]).reshape(1,1)
            
            utterance_batch = np.vstack((utterance_batch,utterance_emb))
            intent_batch = np.vstack((intent_batch,intent_emb))
            slots_batch = np.vstack((slots_batch,slots_emb))
            
            req_batch = np.vstack((req_batch,req_emb))
            
            
            prev_act_batch = np.vstack((prev_act_batch,prev_act_emb))
            prev_intent_batch = np.vstack((prev_intent_batch,prev_intent_emb))
            action_batch = np.vstack((action_batch,action_emb))
            label_batch = np.vstack((label_batch,label))
            
            
            belief_state_batch = np.vstack((belief_state_batch,belief_state_emb))
            iw_batch = np.vstack((iw_batch,iw_emb))
            
            
    
    index_list = [ i for i in range(0, utterance_batch.shape[0] ) ]
    random.shuffle(index_list)
 
    action_batch = np.float64(action_batch) #should change this
   
    utterance_batch = tf.convert_to_tensor(utterance_batch[index_list])
    intent_batch = tf.convert_to_tensor(intent_batch[index_list])
    slots_batch = tf.convert_to_tensor(slots_batch[index_list])
    
    req_batch = tf.convert_to_tensor(req_batch[index_list])
    
    prev_act_batch = tf.convert_to_tensor(prev_act_batch[index_list])
    prev_intent_batch = tf.convert_to_tensor(prev_intent_batch[index_list])
    
    action_batch = tf.convert_to_tensor(action_batch[index_list])
    label_batch = tf.convert_to_tensor(label_batch[index_list])
    
    
    belief_state_batch = tf.convert_to_tensor(belief_state_batch[index_list])
    iw_batch = tf.convert_to_tensor(iw_batch[index_list])
    
    
    
    return utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch












def get_batch(batch_data,frequent,not_frequent):    
    utterance_batch = ""
    intent_batch = ""
    slots_batch = ""
    req_batch = ""
    action_batch = ""
    label_batch = ""
    prev_act_batch = ""
    
    
    belief_state_batch = ""
    iw_batch = ""
    
    
    num_neg_freq = 2
    num_neg_not_freq = 1
    
    
    for data in batch_data:
        utterance_emb = data["utterance_emb"]
        intent_emb = data["intent_emb"]
        slots_emb = data["slots_emb"]
        
        req_emb = data["req_emb"]
        
        
        prev_act_emb = data["prev_action_emb"]
        prev_intent_emb = data["prev_intent_emb"]
        
        
        belief_state_emb = data["belief_state"]
        iw_emb =  data["intent_window_emb"]
        
        
        positive_action = random.sample(data["system_actions"],1)
        positive_emb = dialogue_act_embed[positive_action[0]]
        label = np.array([1]).reshape(1,1)
        
        if utterance_batch =="":
            utterance_batch = utterance_emb
            intent_batch = intent_emb
            slots_batch = slots_emb
            
            req_batch = req_emb
            
            prev_act_batch = prev_act_emb
            action_batch = positive_emb
            prev_intent_batch=prev_intent_emb
            label_batch = label 
            
            
            belief_state_batch = belief_state_emb
            iw_batch = iw_emb
            
            
            
        else:
            utterance_batch = np.vstack((utterance_batch,utterance_emb))
            intent_batch = np.vstack((intent_batch,intent_emb))
            slots_batch = np.vstack((slots_batch,slots_emb))
            
            req_batch = np.vstack((req_batch,req_emb))
            
            prev_act_batch = np.vstack((prev_act_batch,prev_act_emb))
            prev_intent_batch = np.vstack((prev_intent_batch,prev_intent_emb))
            action_batch = np.vstack((action_batch,positive_emb))
            label_batch = np.vstack((label_batch,label))
            
            
            belief_state_batch = np.vstack((belief_state_batch,belief_state_emb))
            iw_batch = np.vstack((iw_batch,iw_emb))
            
        
        
        
        
        negative_frequent = random.sample(list(set(frequent).difference(set(data["system_actions"]))) , num_neg_freq)
        negative_not_frequent = random.sample(list(set(not_frequent).difference(set(data["system_actions"]))) , num_neg_not_freq)
        negative_actions = negative_frequent + negative_not_frequent
        
        
        
        
        
        for action  in negative_actions:
            action_emb = dialogue_act_embed[action]
            label = np.array([0]).reshape(1,1)
            
            utterance_batch = np.vstack((utterance_batch,utterance_emb))
            intent_batch = np.vstack((intent_batch,intent_emb))
            slots_batch = np.vstack((slots_batch,slots_emb))
            
            req_batch = np.vstack((req_batch,req_emb))
            
            
            prev_act_batch = np.vstack((prev_act_batch,prev_act_emb))
            prev_intent_batch = np.vstack((prev_intent_batch,prev_intent_emb))
            action_batch = np.vstack((action_batch,action_emb))
            label_batch = np.vstack((label_batch,label))
            
            
            belief_state_batch = np.vstack((belief_state_batch,belief_state_emb))
            iw_batch = np.vstack((iw_batch,iw_emb))
            
            
    
    index_list = [ i for i in range(0, utterance_batch.shape[0] ) ]
    random.shuffle(index_list)
 
    action_batch = np.float64(action_batch) #should change this
   
    utterance_batch = tf.convert_to_tensor(utterance_batch[index_list])
    intent_batch = tf.convert_to_tensor(intent_batch[index_list])
    slots_batch = tf.convert_to_tensor(slots_batch[index_list])
    
    req_batch = tf.convert_to_tensor(req_batch[index_list])
    
    prev_act_batch = tf.convert_to_tensor(prev_act_batch[index_list])
    prev_intent_batch = tf.convert_to_tensor(prev_intent_batch[index_list])
    
    action_batch = tf.convert_to_tensor(action_batch[index_list])
    label_batch = tf.convert_to_tensor(label_batch[index_list])
    
    
    belief_state_batch = tf.convert_to_tensor(belief_state_batch[index_list])
    iw_batch = tf.convert_to_tensor(iw_batch[index_list])
    
    
    
    return utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch





def get_positive_batch(batch_data):
    utterance_batch = ""
    intent_batch = ""
    slots_batch = ""
    req_batch = ""
    action_batch = ""
    label_batch = ""
    
    belief_state_batch = ""
    iw_batch = ""
    
    
    for data in batch_data:
        utterance_emb = data["utterance_emb"]
        intent_emb = data["intent_emb"]
        slots_emb = data["slots_emb"]
        
        req_emb = data["req_emb"]
        
        prev_intent_emb = data["prev_intent_emb"]
        prev_act_emb = data["prev_action_emb"]
        label = np.array([1]).reshape(1,1)
        
        
        
        belief_state_emb = data["belief_state"]
        iw_emb =  data["intent_window_emb"]
        
        
        
        for action in data["system_actions"]:
            action_emb = dialogue_act_embed[action]
            if utterance_batch =="":
                utterance_batch = utterance_emb
                intent_batch = intent_emb
                slots_batch = slots_emb
                
                req_batch = req_emb
                
                
                prev_act_batch = prev_act_emb
                prev_intent_batch=prev_intent_emb
                
                action_batch = action_emb
                label_batch = label 
                
                
                belief_state_batch = belief_state_emb
                iw_batch = iw_emb
                
                
            else:
                utterance_batch = np.vstack((utterance_batch,utterance_emb))
                intent_batch = np.vstack((intent_batch,intent_emb))
                slots_batch = np.vstack((slots_batch,slots_emb))
                
                
                req_batch = np.vstack((req_batch,req_emb))
                
                
                prev_act_batch = np.vstack((prev_act_batch,prev_act_emb))
                prev_intent_batch = np.vstack((prev_intent_batch,prev_intent_emb))
                
                action_batch = np.vstack((action_batch,action_emb))
                label_batch = np.vstack((label_batch,label))
                
                belief_state_batch = np.vstack((belief_state_batch,belief_state_emb))
                iw_batch = np.vstack((iw_batch,iw_emb))
                
                
                
                
    index_list = [ i for i in range(0, utterance_batch.shape[0]) ]
    #random.shuffle(index_list)

    action_batch = np.float64(action_batch) #should change this
    
    utterance_batch = tf.convert_to_tensor(utterance_batch[index_list])
    intent_batch = tf.convert_to_tensor(intent_batch[index_list])
    slots_batch = tf.convert_to_tensor(slots_batch[index_list])
    
    req_batch = tf.convert_to_tensor(req_batch[index_list])
    
    
    prev_act_batch = tf.convert_to_tensor(prev_act_batch[index_list])
    prev_intent_batch = tf.convert_to_tensor(prev_intent_batch[index_list])
    
    action_batch = tf.convert_to_tensor(action_batch[index_list])
    label_batch = tf.convert_to_tensor(label_batch[index_list])
    
    belief_state_batch = tf.convert_to_tensor(belief_state_batch[index_list])
    iw_batch = tf.convert_to_tensor(iw_batch[index_list])
    
    
    return utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch






def get_single_batch(batch_data):
    
    utterance_batch = ""
    intent_batch = ""
    slots_batch = ""
    req_batch = ""
    action_batch = ""
    label_batch = ""
    prev_act_batch = ""
    belief_state_batch = ""
    iw_batch = ""
    
    
    data = batch_data[0]
    
    
    utterance_emb = data["utterance_emb"]
    intent_emb = data["intent_emb"]
    slots_emb = data["slots_emb"]
    
    req_emb = data["req_emb"]
    
    prev_intent_emb = data["prev_intent_emb"]
    prev_act_emb = data["prev_action_emb"]
#     label = np.array([1]).reshape(1,1)
    belief_state_emb = data["belief_state"]
    iw_emb =  data["intent_window_emb"]


    
    
    all_actions = list(set(dialogue_acts))
    

    for action in all_actions:
        label = ""
        if action in data["system_actions"]:
            label = np.array([1]).reshape(1,1)
        else:
            label = np.array([0]).reshape(1,1)
            
        
        
        action_emb = dialogue_act_embed[action]
        if utterance_batch =="":
            utterance_batch = utterance_emb
            intent_batch = intent_emb
            slots_batch = slots_emb
            
            req_batch = req_emb
            
            prev_act_batch = prev_act_emb
            prev_intent_batch=prev_intent_emb

            action_batch = action_emb
            label_batch = label 


            belief_state_batch = belief_state_emb
            iw_batch = iw_emb


        else:
            utterance_batch = np.vstack((utterance_batch,utterance_emb))
            intent_batch = np.vstack((intent_batch,intent_emb))
            slots_batch = np.vstack((slots_batch,slots_emb))
            
            req_batch = np.vstack((req_batch,req_emb))
            
            
            prev_act_batch = np.vstack((prev_act_batch,prev_act_emb))
            prev_intent_batch = np.vstack((prev_intent_batch,prev_intent_emb))

            action_batch = np.vstack((action_batch,action_emb))
            label_batch = np.vstack((label_batch,label))

            belief_state_batch = np.vstack((belief_state_batch,belief_state_emb))
            iw_batch = np.vstack((iw_batch,iw_emb))
                
                
                
                
    index_list = [ i for i in range(0, utterance_batch.shape[0]) ]
    #random.shuffle(index_list) #should not shuffle here

    action_batch = np.float64(action_batch) #should change this
    
    utterance_batch = tf.convert_to_tensor(utterance_batch[index_list])
    intent_batch = tf.convert_to_tensor(intent_batch[index_list])
    slots_batch = tf.convert_to_tensor(slots_batch[index_list])
    
    req_batch = tf.convert_to_tensor(req_batch[index_list])
    
    
    prev_act_batch = tf.convert_to_tensor(prev_act_batch[index_list])
    prev_intent_batch = tf.convert_to_tensor(prev_intent_batch[index_list])
    
    action_batch = tf.convert_to_tensor(action_batch[index_list])
    label_batch = tf.convert_to_tensor(label_batch[index_list])
    
    belief_state_batch = tf.convert_to_tensor(belief_state_batch[index_list])
    iw_batch = tf.convert_to_tensor(iw_batch[index_list])
    
    
    return utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch
















#sparse categorical
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    name='sparse_categ_crossentropy'
)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='sparse_categ_accuracy', dtype=None
)
train_loss = tf.keras.metrics.Mean(name='train_loss')













Model = ""




def get_metrics(load_model=True):
    
    
    print("Getting Metric")
    dev_data = []
    with open('dev_data.pickle', 'rb') as handle:
        dev_data = pickle.load(handle)

        
    if load_model:    
        print("Model loaded")
        Model = DA_Model()
    #     Model.load_weights('../DA_Prediction/checkpoints/dialogue_act_concat_neg1_dense.ckpt')
        Model.load_weights('./model/model_003')
    
    
    
    
    
    
    complete_label = np.zeros((len(dev_data),347))
    complete_predicted_label = np.zeros((len(dev_data),347))
    print("Started")
    
    
    for rec in range(len(dev_data)):
        
        test_batch=dev_data[rec].copy()
        batch_data = [test_batch]
#         print(batch_data)
#         break
        utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch=get_single_batch(batch_data)
    
        predictions = Model(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,prev_act_batch, belief_state_batch,iw_batch,req_batch,False)
        
        
        
        predictions = tf.nn.softmax(predictions)
        predicted_label = (predictions[:,1].numpy()>0.92)+0  #check this
        
        
        
        label = label_batch.numpy().reshape(-1,347)
        predicted_label = predicted_label.reshape(-1,347)

        
        if rec%100==0:
            print(rec)
        
#         if complete_label =="":
#             complete_label = label
#             complete_predicted_label = predicted_label
#         else:
#             complete_label = np.vstack((complete_label,label))
#             complete_predicted_label = np.vstack((complete_predicted_label,predicted_label))
        
                                        
                                        
        complete_label[rec]  =  label
        complete_predicted_label[rec] = predicted_label
                  
            
        if rec%100==0:
            print(rec)    
            
        if rec>1000 and False:  #check this
            print(rec)
            complete_label = complete_label[0:rec+1,:]
            complete_predicted_label = complete_predicted_label[0:rec+1,:]
            break
        
        
    
    print(complete_label.shape)
    print(complete_predicted_label.shape)
    
#     print(complete_label)
#     print(complete_predicted_label)
    
    
#     import warnings
#     warnings.filterwarnings('always')
    
    present_true = list(set(list(np.where(complete_label==1)[1])))
    present_predicted = list(set(list(np.where(complete_predicted_label==1)[1])))
    present_predicted = np.array(list(set(present_true+present_predicted)))
#     present_predicted = None
#     print(present_predicted)
    
    
    from sklearn.metrics import f1_score,recall_score,precision_score
    score = f1_score(y_true= complete_label, y_pred=complete_predicted_label, average='weighted',labels=present_predicted)
    print(score)
    print(recall_score(y_true= complete_label, y_pred=complete_predicted_label, average='weighted',labels=present_predicted))
    print(precision_score(y_true= complete_label, y_pred=complete_predicted_label,average='weighted',labels=present_predicted))
    
    







def eval_model(load_model=True, positive_batch = True,is_ratio=False):
    

    
    global Model
    
    print("Evaluating the Model")
    dev_data = []
    with open('dev_data.pickle', 'rb') as handle:
        dev_data = pickle.load(handle)

        
        
    if load_model:      
        print("Model loaded")
        Model = DA_Model()
    #     Model.load_weights('../DA_Prediction/checkpoints/dialogue_act_concat_neg1_dense.ckpt')
        Model.load_weights('./model/model_003')
    
    
    print("Started")
    dev_batches = len(dev_data)//32
    print("Number of dev batches: ",dev_batches)
    shuffled_datapoints = random.sample(dev_data,dev_batches*32)
    dev_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='sparse_categ_accuracy_dev', dtype=None
    )
    dev_accuracy.reset_states()
    
    
    
    
    frequent = []
    not_frequent = []
    if not positive_batch:
        coverage = {}
        for num,i in enumerate(dev_data):    
            actions = i["system_actions"]
            for action in actions:
                if action in coverage:
                    coverage[action]+=1
                else:
                    coverage[action] = 1
        coverage = {k: v for k, v in sorted(coverage.items(), key=lambda item: item[1])}
        not_frequent = []
        frequent = []
        for action in coverage:
            if coverage[action]<300:
                not_frequent.append(action)
            else:
                frequent.append(action)
        print("Total number of frequent actions are:",len(frequent))
        print("Total number of Non frequent actionsa are:", len(not_frequent))
        
    
    
    
    
    for batch_num in range(dev_batches):
        start_num = batch_num*32
        end_num = start_num+32
        batch_data = shuffled_datapoints[start_num:end_num]
        
        
       
        if positive_batch:
             utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch = get_positive_batch(batch_data)

        else:
            if is_ratio:
                utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch = get_batch_ratio(batch_data,frequent,not_frequent)
                
            else:
                utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch = get_batch(batch_data,frequent,not_frequent)
            
           

        
        if batch_num==0:
            print(utterance_batch.numpy().shape)
        
        
        predictions = Model(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,prev_act_batch, belief_state_batch,iw_batch,req_batch,False)
        

            
#         predictions = Model(intent_batch,slots_batch,req_batch,action_batch,utterance_batch,prev_act_batch,False)
#         predicted_label = np.argmax(predictions.numpy(),axis=-1)
#         local_acc = (predicted_label.reshape(-1,1) == label_batch.numpy()).all(axis=(-1)).mean()




        dev_accuracy.update_state(label_batch, predictions)
        if batch_num%400==0 and batch_num!=0:
            print ('Batch {}  Accuracy {:.4f}'.format(
          batch_num, dev_accuracy.result()))
            
            
    print("Final dev accuracy: ",dev_accuracy.result())
    
    
    return dev_accuracy.result()











mode = sys.argv[1]
if mode=="train":    
    
    
    
    train_accuracies = []
    dev_accuracies = []
    
    print("Training the Model")
    
    training_data=[]  #total datapoints 164982
    with open('training_data.pickle', 'rb') as handle:
        training_data = pickle.load(handle)
    Model = DA_Model()
    
    
    
    Epochs = 80
    no_of_batches = (len(training_data)//32)
    print("Started")
    
    
    
    
    
    coverage = {}
    for num,i in enumerate(training_data):    
        actions = i["system_actions"]
        for action in actions:
            if action in coverage:
                coverage[action]+=1
            else:
                coverage[action] = 1
    coverage = {k: v for k, v in sorted(coverage.items(), key=lambda item: item[1])}
    not_frequent = []
    frequent = []
    for action in coverage:
        if coverage[action]<1000:
            not_frequent.append(action)
        else:
            frequent.append(action)
    print("Total number of frequent actions are:",len(frequent))
    print("Total number of Non frequent actionsa are:", len(not_frequent))
    
    
    
    
    
    
    
    
    
    is_ratio = False
    print("Is_raio:-",is_ratio)
    for epoch in range(0,Epochs):
        print("Epoch Num:", epoch+1)
        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        shuffled_datapoints = random.sample(training_data,no_of_batches*32)

        for batch_num in range(no_of_batches):
            start_num = batch_num*32
            end_num = start_num+32
            batch_data = shuffled_datapoints[start_num:end_num]
            
            
            
            
            if is_ratio:
                utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch = get_batch_ratio(batch_data,frequent,not_frequent)
                
            else:
                utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch = get_batch(batch_data,frequent,not_frequent)

            train_step(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch,req_batch)

            
            
            
            
            if batch_num==0:
                print(utterance_batch.numpy().shape)
            
            
            if batch_num%1000==0 and batch_num!=0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch_num, train_loss.result(), train_accuracy.result()))
#             break
        
    
    
        print('Time taken for one epoch {} secs\n'.format(time.time() - start_time)) #takes 125seconds       
        
        
        
        temp_dev_accuracy = ""
        if is_ratio:
            temp_dev_accuracy_positive = eval_model(load_model=False)
            temp_dev_accuracy_negative = eval_model(load_model=False,positive_batch=False,is_ratio=is_ratio)
            temp_dev_accuracy = (0.65*temp_dev_accuracy_positive) + (0.35*temp_dev_accuracy_negative)

        else:
            temp_dev_accuracy_positive = eval_model(load_model=False)
            temp_dev_accuracy_negative = eval_model(load_model=False,positive_batch=False)
            temp_dev_accuracy = (0.79*temp_dev_accuracy_positive) + (0.21*temp_dev_accuracy_negative)
        
        
        
        
        dev_accuracies.append( float(temp_dev_accuracy) )
        train_accuracies.append( float(train_accuracy.result())  )
        
        if epoch<3 or temp_dev_accuracy >= max(dev_accuracies[3:]) :
            print("Saving Weights")
            Model.save_weights('./model/model_003')
        print(train_accuracies)
        print(dev_accuracies)
        
        
        
        
        
        
elif mode=="metric":
    get_metrics()
        
        
        
        
else:
    eval_model()






























































"""
def eval_model():
    print("Evaluating the Model")
    dev_data = []
    with open(dev_pickle, 'rb') as handle:
        dev_data = pickle.load(handle)


    Model = DA_Model()
    #Model.load_weights(model_weight_path)
    Model.load_weights(data_path+'model/model_003')


    dialogue_dict={}
    for id, val in enumerate(dialogue_acts):
        dialogue_dict[id]=val




    print ("dialogue_acts:")
    print (dialogue_acts)
    batch_size=16
    print("Started")
    dev_batches = len(dev_data)//batch_size
    
    
    
    print("Number of dev batches: ",dev_batches)
    dict_resp={}
    count = 0
    
    
    for rec in range(len(dev_data)):
        test_batch=dev_data[rec].copy()
        test_batch['system_actions']=dialogue_acts

        batch_data = [test_batch]
        
        utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch = get_positive_batch(batch_data)
        
        predictions = Model(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,prev_act_batch, belief_state_batch,iw_batch,False)

        pred=predictions.numpy()
        dict_resp={}
        idval=np.argmax(pred[:,1])
        value=dialogue_dict[idval]
        #print(pred)
        #print (dev_data[rec]['system_actions'], value)
        if dev_data[rec]['system_actions']=={value}:
                count += 1
        else:
                print (value,dev_data[rec]['system_actions'])
    print ("Accuracy:",count/(rec+1))



    
    
mode = sys.argv[1]
if mode=="train":
    
    print("Training the Model")
    training_data=[]  #total datapoints 164982
    with open(train_pickle, 'rb') as handle:
        training_data = pickle.load(handle)
    Model = DA_Model()

    Epochs = 80
    batch_size=16


    no_of_batches = (len(training_data)//batch_size)
    print ("no_of_batches:",no_of_batches)
    print("Started")


    max_accuracy = 0


    for epoch in range(0,Epochs):
        print("Epoch Num:", epoch+1)
        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        shuffled_datapoints = random.sample(training_data,no_of_batches*batch_size)

        for batch_num in range(no_of_batches):
            start_num = batch_num*batch_size
            end_num = start_num+batch_size
            batch_data = shuffled_datapoints[start_num:end_num]
            #actual_action=batch_data[0]['system_actions']
            #print (actual_action)
            #batch_data[0]['system_actions']=dialogue_acts



            utterance_batch,intent_batch,slots_batch,prev_intent_batch,action_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch = get_batch(batch_data)
            
            train_step(intent_batch,slots_batch,prev_intent_batch,action_batch,utterance_batch,label_batch,prev_act_batch,belief_state_batch,iw_batch)




            if batch_num%35==0 and batch_num!=0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                      epoch + 1, batch_num, train_loss.result(), train_accuracy.result()))


    #     if train_accuracy.result()>max_accuracy:
    #         max_accuracy = train_accuracy.result()
    #         print("saving model with accuracy:",max_accuracy)
    #         Model.save_weights(data_path+'model/model_003')

        print('Time taken for one epoch {} secs\n'.format(time.time() - start_time)) #takes 125seconds 
    Model.save_weights(data_path+'model/model_003')
    eval_model()

 
else:
    eval_model()
"""

