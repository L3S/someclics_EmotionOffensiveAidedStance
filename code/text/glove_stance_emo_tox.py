import tensorflow as tf
from keras import backend as K
from keras import backend as k
from keras import layers
from keras.layers.core import Lambda
import numpy as np
from numpy import array 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dropout,Bidirectional,Input, Embedding, Dense,Concatenate,Flatten, Multiply,Average,Subtract
from keras.models import Model
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.engine import Layer
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras import optimizers,regularizers
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import multilabel_confusion_matrix
import ast

def expandDim(x):
    x=k.expand_dims(x, 1)
    return x

def linearConv(var):
    ten1,ten2=var[0],var[1]
    ten1=expandDim(ten1)
    # print("ten1************",ten1)
    ten2=expandDim(ten2)
    # print("ten2************",ten2)
    arr1=tf.nn.conv1d(ten2, ten1, padding='SAME', stride=1)
    arr1=tf.squeeze(arr1,axis=1)
    print("arr1####:::::",arr1)
    return arr1

def attentionScores(var):
    Q_t,K_t,V_t=var[0],var[1],var[2]
    scores = tf.matmul(Q_t, K_t, transpose_b=True)
    print("first scores shape:::::",scores.shape)
    distribution = tf.nn.softmax(scores)
    scores=tf.matmul(distribution, V_t)
    print("scores shape:::::",scores.shape)
    return scores

def create_resample(train_sequence,train_stance_enc,train_emo,train_toxic):
    df = pd.DataFrame(list(zip(train_sequence,train_stance_enc,train_emo,train_toxic)), columns =['text','stance','emo','toxic'],index=None)
    ambi=(df[df['stance'] == 0])
    print("len ambi",len(ambi))
    blv=(df[df['stance'] == 1])
    print("len blv",len(blv))
    deny=(df[df['stance'] == 2])
    print("len deny",len(deny))

    upsampled1 = resample(ambi,replace=True, # sample with replacement
                          n_samples=len(blv), # match number in majority class
                          random_state=27)
    upsampled2 = resample(deny,replace=True, # sample with replacement
                          n_samples=len(blv), # match number in majority class
                          random_state=27)

    upsampled = pd.concat([blv,upsampled1,upsampled2])
    upsampled=upsampled.sample(frac=1)
    print("After oversample train data : ",len(upsampled))
    print("After oversampling, instances of tweet act classes in oversampled data :: ",upsampled.stance.value_counts())

    train_data=upsampled
    train_sequence=[]
    train_stance_enc=[]
    train_emo=[]
    train_toxic=[]

   
    for i in range(len(train_data)):
        train_sequence.append(train_data.text.values[i])
        train_stance_enc.append(train_data.stance.values[i])
        train_emo.append(train_data.emo.values[i])
        train_toxic.append(train_data.toxic.values[i])


    return train_sequence,train_stance_enc,train_emo,train_toxic



data=pd.read_csv("../../final_data.csv", delimiter=";", na_filter= False) 
print("data :: ",len(data))


########### creating multiple lists as per the daataframe

li_text=[]
li_stance=[]
li_emo1=[]
li_id=[]
li_toxic=[]
li_emo=[]
li_toxic1=[]
li_toxic=[]

for i in range(len(data)):
    li_id.append(data.tweetid.values[i])
    li_stance.append(data.stance.values[i])
    li_text.append((data.text.values[i]))
    li_emo1.append((data.emotion.values[i]))
    li_toxic1.append((data.toxic_multi.values[i]))


for i in range(len(li_emo1)):
    x=li_emo1[i]
    x=ast.literal_eval(x)
    y=[]
    for j in x:
        j=int(j)
        y.append(j)
    li_emo.append(y)
    x=li_toxic1[i]
    x=ast.literal_eval(x)
    y=[]
    for j in x:
        j=int(j)
        y.append(j)
    li_toxic.append(y)

print("li_stance np unique:::",np.unique(li_stance,return_counts=True))
print("li_emo np unique:::",np.unique(li_emo,return_counts=True))
print("li_toxic np unique:::",np.unique(li_toxic,return_counts=True))




########### converting labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=li_stance
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)

label_enc=total_integer_encoded

label_cat=to_categorical(total_integer_encoded)

label_emo=np.array(li_emo)
label_toxic=np.array(li_toxic)

########## converting text modality into sequence of vectors ############
total_text= li_text
total_text = [x.lower() for x in total_text] 

MAX_SEQ=50

tokenizer = Tokenizer()
tokenizer.fit_on_texts(total_text)
total_sequence = tokenizer.texts_to_sequences(total_text)
padded_docs = pad_sequences(total_sequence, maxlen=MAX_SEQ, padding='post')

total_sequence=padded_docs

vocab_size = len(tokenizer.word_index) + 1

# print("downloading ###")
embedding_matrix= np.load("../../embedding_matrix/embed_matrix_glove.npy")
print("embedding matrix ****************",embedding_matrix.shape)

MAX_LENGTH=50

#######data for K-fold #########

list_acc_stance,list_f1_stance,list_prec_stance,list_rec_stance=[],[],[],[]
list_acc_emo,list_f1_emo,list_prec_emo,list_rec_emo=[],[],[],[]
list_acc_tox,list_f1_tox,list_prec_tox,list_rec_tox=[],[],[],[]


kf=StratifiedKFold(n_splits=5, random_state=None,shuffle=False)
fold=0
results=[]
for train_index,test_index in kf.split(total_sequence,label_enc):
    print("K FOLD ::::::",fold)
    fold=fold+1

    ############## Shared input #############

    input_shared = Input (shape = (MAX_LENGTH, ))
    input_text_shared = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=50, name='text_embed_share')(input_shared)
    
    #### stance #######
    lstm_stance = Bidirectional(LSTM(100, name='lstm_stance', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_shared)
    Q_s= Dense(100, activation="relu")(lstm_stance)
    K_s= Dense(100, activation="relu")(lstm_stance)
    V_s= Dense(100, activation="relu")(lstm_stance)
    IA_stance=Lambda(attentionScores)([Q_s,K_s,V_s])
    
    #### emotion #######
    lstm_emo = Bidirectional(LSTM(100, name='lstm_emo', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_shared)
    Q_e= Dense(100, activation="relu")(lstm_emo)
    K_e= Dense(100, activation="relu")(lstm_emo)
    V_e= Dense(100, activation="relu")(lstm_emo)
    IA_emo=Lambda(attentionScores)([Q_e,K_e,V_e])

    emo_output=Dense(10, activation="sigmoid", name="task_emo")(IA_emo)

    LCM_output_emo=Lambda(linearConv)([IA_stance,emo_output])
    print("LCM_output_emo:::",LCM_output_emo)

    #####toxic ######
    lstm_toxic = Bidirectional(LSTM(100, name='lstm_emo', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(input_text_shared)
    Q_t= Dense(100, activation="relu")(lstm_toxic)
    K_t= Dense(100, activation="relu")(lstm_toxic)
    V_t= Dense(100, activation="relu")(lstm_toxic)

    IA_toxic=Lambda(attentionScores)([Q_t,K_t,V_t])

    toxic_output=Dense(8, activation="sigmoid", name="task_toxic")(IA_toxic)
    reshape_toxic = Dense(100, activation="relu")(toxic_output)

    LCM_output_toxic=Lambda(linearConv)([IA_stance,reshape_toxic])
    print("LCM_output_toxic:::",LCM_output_toxic)

    #### shared output ##############
    shared_input=Average()([IA_stance,IA_emo,IA_toxic])
    shared_output= Dense(100, activation="relu")(shared_input)

    ###### emotion output #######
    feat_emo= Concatenate()([shared_output,IA_emo])
    emo_output=Dense(10, activation="sigmoid", name="task_emo")(feat_emo)
    reshape_emo = Dense(100, activation="relu")(emo_output)
    LCM_output_emo=Lambda(linearConv)([IA_stance,reshape_emo])
    print("LCM_output_emo:::",LCM_output_emo)

    ###### toxic output ##########
    feat_toxic= Concatenate()([shared_output,IA_toxic])
    toxic_output=Dense(8, activation="sigmoid", name="task_toxic")(feat_toxic)
    reshape_toxic = Dense(100, activation="relu")(toxic_output)
    LCM_output_toxic=Lambda(linearConv)([IA_stance,reshape_toxic])
    print("LCM_output_toxic:::",LCM_output_toxic)

    ######### stance specific shared output ############
    K_sh= Dense(100, activation="relu")(shared_output)
    V_sh= Dense(100, activation="relu")(shared_output)
    Q_ss= Dense(100, activation="relu")(IA_stance)

    ######### stance specific shared output ############
    K_sh= Dense(100, activation="relu")(shared_output)
    V_sh= Dense(100, activation="relu")(shared_output)

    SAM_output=Lambda(attentionScores)([Q_s,K_sh,V_sh])
    print("SAM_output:::",SAM_output)

    ########### Integration of LCM and SAM ###########
    LCM_output=Average()([LCM_output_emo,LCM_output_toxic])
    
    int_diff=layers.subtract([LCM_output,SAM_output])
    intg_mul=Multiply()([LCM_output,SAM_output])

    IM_output=Concatenate()([LCM_output,SAM_output,int_diff,intg_mul])


    stance_output=Dense(3, activation="softmax", name="task_stance")(IM_output)
    print("stance_output:::::",stance_output)
  
    model=Model([input_shared],[stance_output,emo_output,toxic_output])


    ##Compile
    model.compile(optimizer=Adam(0.001),loss={'task_stance':'categorical_crossentropy','task_emo':'binary_crossentropy','task_toxic':'binary_crossentropy'},
    loss_weights={'task_stance':1.0,'task_emo':0.5,'task_toxic':0.3}, metrics=['accuracy'])    
    print(model.summary())

    #### model fit ############
    test_sequence,train_sequence=total_sequence[test_index],total_sequence[train_index]
    
    test_stance_enc,train_stance_enc=label_enc[test_index],label_enc[train_index]
    test_emo,train_emo=label_emo[test_index],label_emo[train_index]
    test_toxic,train_toxic=label_toxic[test_index],label_toxic[train_index]
    
    print("len of train",np.unique(train_stance_enc,return_counts=True),len(train_stance_enc))
    print("len of test",np.unique(test_stance_enc,return_counts=True),len(test_stance_enc))

    train_sequence,train_stance_enc,train_emo,train_toxic=create_resample(train_sequence,train_stance_enc,train_emo,train_toxic)

    train_sequence=np.array(train_sequence)
    train_emo=np.array(train_emo)
    test_emo=np.array(test_emo)
    train_toxic=np.array(train_toxic)
    test_toxic=np.array(test_toxic)

    train_stance=to_categorical(train_stance_enc)


    model.fit([train_sequence],[train_stance,train_emo,train_toxic], shuffle=True,validation_split=0.2,epochs=20,verbose=2)
    predicted = model.predict([test_sequence])
    print(predicted)

    stance_specific=predicted[0]
    result_=stance_specific
    p_1 = np.argmax(result_, axis=1)
    test_accuracy=accuracy_score(test_stance_enc, p_1)
    list_acc_stance.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names = ['ambiguous','believe','deny']
    class_rep=classification_report(test_stance_enc, p_1)
    print("specific confusion matrix",confusion_matrix(test_stance_enc, p_1))
    print(class_rep)
    class_rep=classification_report(test_stance_enc, p_1, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    macro_prec=class_rep['macro avg']['precision']
    macro_rec=class_rep['macro avg']['recall']
    print("macro f1 score",macro_avg)
    list_f1_stance.append(macro_avg)
    list_prec_stance.append(macro_prec)
    list_rec_stance.append(macro_rec)


    ########### emotion performance ##########
    emo_specific=predicted[1]
    result_=emo_specific
    y_pred=[]
    for sample in  result_:
        sample=list(sample)
        mean_=statistics.mean(sample)

        y_pred.append([1 if i>=mean_ else 0 for i in sample])

    y_pred = np.array(y_pred)

    test_accuracy=accuracy_score(test_emo,y_pred)
    list_acc_emo.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names =['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',"positive","negative"]
    class_rep=classification_report(test_emo, y_pred)
    print("specific confusion matrix",multilabel_confusion_matrix(test_emo, y_pred))
    print(class_rep)
    class_rep=classification_report(test_emo, y_pred, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    micro_avg=class_rep['micro avg']['f1-score']
    macro_prec=class_rep['micro avg']['precision']
    macro_rec=class_rep['micro avg']['recall']
    print("macro f1 score",macro_avg)
    list_f1_emo.append(macro_avg)
    list_prec_emo.append(macro_prec)
    list_rec_emo.append(macro_rec)

    ########### toxic performance ##########
    toxic_specific=predicted[2]
    result_=toxic_specific
    y_pred=[]
    for sample in  result_:
        sample=list(sample)
        mean_=statistics.mean(sample)

        y_pred.append([1 if i>=mean_ else 0 for i in sample])

    y_pred = np.array(y_pred)

    test_accuracy=accuracy_score(test_toxic,y_pred)
    list_acc_tox.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names =['SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT','SEXUALLY_EXPLICIT','TOXICITY','NON_TOXIC']
    class_rep=classification_report(test_toxic, y_pred)
    print("specific confusion matrix",multilabel_confusion_matrix(test_toxic, y_pred))
    print(class_rep)
    class_rep=classification_report(test_toxic, y_pred, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    micro_avg=class_rep['micro avg']['f1-score']
    macro_prec=class_rep['micro avg']['precision']
    macro_rec=class_rep['micro avg']['recall']
    print("macro f1 score",macro_avg)
    list_f1_tox.append(macro_avg)
    list_prec_tox.append(macro_prec)
    list_rec_tox.append(macro_rec)

model_architecture = model.to_json()
with open('model/glove_stance_emo_toxic_final.json', 'w') as json_file:
    json_file.write(model_architecture)


model.save_weights('model/glove_stance_emo_toxic_final_weights.h5')


    
############# stance 
print("Stance ::::::::::::::::::::::")
print("ACCURACY :::::::::::: #############")
print("Accuracy  ::: ",list_acc_stance)
print("Mean, STD DEV", statistics.mean(list_acc_stance),statistics.stdev(list_acc_stance))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_stance)
print("MTL Mean, STD DEV", statistics.mean(list_f1_stance),statistics.stdev(list_f1_stance))


print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_stance)
print("MTL Mean, STD DEV", statistics.mean(list_prec_stance),statistics.stdev(list_prec_stance))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_stance)
print("MTL Mean, STD DEV", statistics.mean(list_rec_stance),statistics.stdev(list_rec_stance))


############# emotion 
print("Emotion ::::::::::::::::::::::")
print("ACCURACY :::::::::::: #############")
print("Accuracy  ::: ",list_acc_emo)
print("Mean, STD DEV", statistics.mean(list_acc_emo),statistics.stdev(list_acc_emo))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_emo)
print("MTL Mean, STD DEV", statistics.mean(list_f1_emo),statistics.stdev(list_f1_emo))


print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_emo)
print("MTL Mean, STD DEV", statistics.mean(list_prec_emo),statistics.stdev(list_prec_emo))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_emo)
print("MTL Mean, STD DEV", statistics.mean(list_rec_emo),statistics.stdev(list_rec_emo))


############# Toxic 
print("Toxic ::::::::::::::::::::::")
print("ACCURACY :::::::::::: #############")
print("Accuracy  ::: ",list_acc_tox)
print("Mean, STD DEV", statistics.mean(list_acc_tox),statistics.stdev(list_acc_tox))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_tox)
print("MTL Mean, STD DEV", statistics.mean(list_f1_tox),statistics.stdev(list_f1_tox))


print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_tox)
print("MTL Mean, STD DEV", statistics.mean(list_prec_tox),statistics.stdev(list_prec_tox))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_tox)
print("MTL Mean, STD DEV", statistics.mean(list_rec_tox),statistics.stdev(list_rec_tox))