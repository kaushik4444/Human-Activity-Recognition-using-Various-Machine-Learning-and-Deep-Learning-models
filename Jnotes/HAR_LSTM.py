#!/usr/bin/env python
# coding: utf-8

# ### NISHANTH IRUTHAYARAJ
# Matric. No: 1522044
# 

# **Basketabll Activity Recognition using KNN Classification Algorithm and LSTM Neural Network**

# **Data Preprocessing**

# In[1]:


import pandas as pd
import numpy as np

#Reading files and Correcting Timestamp#
c_heads = ['Time', 'Device', 'e1' , 'e2' , 'e3' , 'e4' , 'e5' , 'e6' ,'Label']
df_sub1_esense = pd.read_csv('eSense_sbj1.csv', names = c_heads)
df_sub1_esense['Time'] = pd.to_datetime(df_sub1_esense['Time'], unit = 'ms') + pd.Timedelta('02:00:00')
df_sub1_esense = df_sub1_esense.set_index('Time').resample('10L', offset = 4000000).first().interpolate()
df_sub1_esense = df_sub1_esense.sort_index()

df_sub2_esense = pd.read_csv('eSense_sbj2.csv', names = c_heads)
df_sub2_esense['Time'] = pd.to_datetime(df_sub2_esense['Time'], unit = 'ms') + pd.Timedelta('02:00:00')
df_sub2_esense = df_sub2_esense.set_index('Time').resample('10L', offset = 4000000).first().interpolate()
df_sub2_esense = df_sub2_esense.sort_index()


# In[2]:


#Data Interpolating and Upsampling
#Subject 1 ankle
df_layup1_ankle = pd.read_csv('layup_sbj1_ankle.csv', index_col = 0, comment= ';')
df_layup1_ankle.index = pd.date_range(start = '2021-04-14 18:22:13.554', periods = df_layup1_ankle.shape[0], freq = "10.869565L")
df_layup1_ankle = df_layup1_ankle.resample('10L', offset = 4000000).first().interpolate()

df_movements1_ankle_1 = pd.read_csv('movements_sbj1_ankle_1.csv', index_col = 0, comment= ';')
df_movements1_ankle_1.index = pd.date_range(start = '2021-04-14 18:42:32.024', periods = df_movements1_ankle_1.shape[0], freq = "10.869565L")
df_movements1_ankle_1 = df_movements1_ankle_1.resample('10L', offset = 4000000).first().interpolate()

df_shooting1_ankle_1 = pd.read_csv('shooting_sbj1_ankle.csv', index_col = 0, comment= ';')
df_shooting1_ankle_1.index = pd.date_range(start = '2021-04-14 18:12:54.024', periods = df_shooting1_ankle_1.shape[0], freq = "10.869565L")
df_shooting1_ankle_1 = df_shooting1_ankle_1.resample('10L', offset = 4000000).first().interpolate()

df_sub1_ankle = pd.concat([df_layup1_ankle,df_movements1_ankle_1,df_shooting1_ankle_1])
df_sub1_ankle = df_sub1_ankle.set_axis(['a1', 'a2', 'a3'], axis = 'columns')
#df_sub1_ankle = df_sub1_ankle.sort_index()

#Subject 2 ankle
df_dribbling2_ankle = pd.read_csv('dribbling_sbj2_ankle.csv', index_col = 0, comment= ';')
df_dribbling2_ankle.index = pd.date_range(start = '2021-04-14 19:26:42.914', periods = df_dribbling2_ankle.shape[0], freq = "10.869565L")
df_dribbling2_ankle = df_dribbling2_ankle.resample('10L', offset = 4000000).first().interpolate()

df_layup2_ankle = pd.read_csv('layup_sbj2_ankle.csv', index_col = 0, comment= ';')
df_layup2_ankle.index = pd.date_range(start = '2021-04-14 19:16:25.794', periods = df_layup2_ankle.shape[0], freq = "10.869565L")
df_layup2_ankle = df_layup2_ankle.resample('10L', offset = 4000000).first().interpolate()

df_movements2_ankle_1 = pd.read_csv('movements_sbj2_ankle_1.csv', index_col = 0, comment= ';') 
df_movements2_ankle_1.index = pd.date_range(start = '2021-04-14 19:36:19.244', periods = df_movements2_ankle_1.shape[0], freq = "10.869565L")
df_movements2_ankle_1 = df_movements2_ankle_1.resample('10L', offset = 4000000).first().interpolate()

df_shooting2_ankle_1 = pd.read_csv('shooting_sbj2_ankle_1.csv', index_col = 0, comment= ';')
df_shooting2_ankle_1.index = pd.date_range(start = '2021-04-14 18:55:50.364', periods = df_shooting2_ankle_1.shape[0], freq = "10.869565L")
df_shooting2_ankle_1 = df_shooting2_ankle_1.resample('10L', offset = 4000000).first().interpolate()

df_sub2_ankle = pd.concat([df_dribbling2_ankle, df_layup2_ankle, df_movements2_ankle_1, df_shooting2_ankle_1])
df_sub2_ankle = df_sub2_ankle.set_axis(['a1','a2','a3'], axis = 'columns')
#df_sub2_ankle = df_sub2_ankle.sort_index()


# In[3]:


#Subject 1 wrist
df_layup1_wrist = pd.read_csv('layup_sbj1_wrist.csv', index_col = 0, comment= ';')
df_layup1_wrist.index = pd.date_range(start = '2021-04-14 18:22:52.524', periods = df_layup1_wrist.shape[0], freq = "10.869565L")
df_layup1_wrist = df_layup1_wrist.resample('10L', offset = 4000000).first().interpolate()

df_movements1_wrist = pd.read_csv('movements_sbj1_wrist.csv', index_col = 0, comment= ';')
df_movements1_wrist.index = pd.date_range(start = '2021-04-14 18:42:05.244', periods = df_movements1_wrist.shape[0], freq = "10.869565L")
df_movements1_wrist = df_movements1_wrist.resample('10L', offset = 4000000).first().interpolate()

df_shooting1_wrist = pd.read_csv('shooting_sbj1_wrist.csv', index_col = 0, comment= ';')
df_shooting1_wrist.index = pd.date_range(start = '2021-04-14 18:13:12.834', periods = df_shooting1_wrist.shape[0], freq = "10.869565L")
df_shooting1_wrist = df_shooting1_wrist.resample('10L', offset = 4000000).first().interpolate()

df_sub1_wrist = pd.concat([df_layup1_wrist, df_movements1_wrist, df_shooting1_wrist])
df_sub1_wrist = df_sub1_wrist.set_axis(['w1','w2','w3'], axis = 'columns')
#df_sub1_wrist = df_sub1_wrist.sort_index()

#Subject 2 wrist
df_dribbling2_wrist = pd.read_csv('dribbling_sbj2_wrist.csv', index_col = 0, comment= ';')
df_dribbling2_wrist.index = pd.date_range(start = '2021-04-14 19:26:24.464', periods = df_dribbling2_wrist.shape[0], freq = "10.869565L")
df_dribbling2_wrist = df_dribbling2_wrist.resample('10L',  offset = 4000000).first().interpolate()

df_layup2_wrist = pd.read_csv('layup_sbj2_wrist.csv', index_col = 0, comment= ';')
df_layup2_wrist.index = pd.date_range(start = '2021-04-14 19:16:05.604', periods = df_layup2_wrist.shape[0], freq = "10.869565L")
df_layup2_wrist = df_layup2_wrist.resample('10L',  offset = 4000000).first().interpolate()

df_movements2_wrist = pd.read_csv('movements_sbj2_wrist.csv', index_col = 0, comment= ';') 
df_movements2_wrist.index = pd.date_range(start = '2021-04-14 19:36:45.074', periods = df_movements2_wrist.shape[0], freq = "10.869565L")
df_movements2_wrist = df_movements2_wrist.resample('10L',  offset = 4000000).first().interpolate()

df_shooting2_wrist = pd.read_csv('shooting_sbj2_wrist.csv', index_col = 0, comment= ';')
df_shooting2_wrist.index = pd.date_range(start = '2021-04-14 18:56:21.474', periods = df_shooting2_wrist.shape[0], freq = "10.869565L")
df_shooting2_wrist = df_shooting2_wrist.resample('10L',  offset = 4000000).first().interpolate()

df_sub2_wrist = pd.concat([df_dribbling2_wrist, df_layup2_wrist, df_movements2_wrist, df_shooting2_wrist])
df_sub2_wrist = df_sub2_wrist.set_axis(['w1','w2','w3'], axis = 'columns')
#df_sub2_wrist = df_sub2_wrist.sort_index()


# In[4]:


#Merging and Normalising each 
df_sub1 = df_sub1_esense.merge(df_sub1_ankle, left_index=True, right_index=True)
df_1 = df_sub1.merge(df_sub1_wrist, left_index=True, right_index=True)
df_1 = (df_1 - df_1.mean()) / df_1.std()
df_1 = df_1.drop(['Device'], axis = 1)

df_sub2 = df_sub2_esense.merge(df_sub2_ankle, left_index=True, right_index=True)
df_2 = df_sub2.merge(df_sub2_wrist, left_index=True, right_index=True)
df_2 = (df_2 - df_2.mean()) / df_2.std()
df_2 = df_2.drop(['Device'], axis = 1)


# In[5]:


#Finding Samples for Time Intersection
intersection_1 = [df_1.index.isin(df_1.between_time('18:23:14', '18:25:28').index),
                  df_1.index.isin(df_1.between_time('18:45:17', '18:47:16').index),
                  df_1.index.isin(df_1.between_time('18:13:25', '18:15:47').index),
                  df_1.index.isin(df_1.between_time('18:42:48', '18:45:17').index)]
activity_1 = ['layup', 'running', 'shooting', 'walking']
df_1['Label'] = np.select(intersection_1, activity_1, 'Null')
df_1['Subjects'] = 'Subject1'
df_1 = df_1[['e1','e2','e3','e4','e5','e6','a1','a2','a3','w1','w2','w3','Label','Subjects']]

intersection_2 = [df_2.index.isin(df_2.between_time('19:26:58', '19:29:59').index), 
                  df_2.index.isin(df_2.between_time('19:17:35', '19:20:14').index),
                  df_2.index.isin(df_2.between_time('19:39:47', '19:41:55').index),
                  df_2.index.isin(df_2.between_time('19:00:14', '19:02:12').index),
                  df_2.index.isin(df_2.between_time('19:37:15', '19:39:46').index)]
activity_2 = ['dribbling', 'layup', 'running', 'shooting', 'walking']
df_2['Label'] = np.select(intersection_2, activity_2, 'Null')
df_2['Subjects'] = 'Subject2'
df_2 = df_2[['e1','e2','e3','e4','e5','e6','a1','a2','a3','w1','w2','w3','Label','Subjects']]

df = pd.concat([df_1, df_2])
df.reset_index(drop=True, inplace=True)
df.to_csv('df_activity.csv')


# In[6]:


#Checking total number of data for each class 
null = df[df['Label'].str.contains('Null')]
dribbling = df[df['Label'].str.contains('dribbling')]
layup = df[df['Label'].str.contains('layup')]
running = df[df['Label'].str.contains('running')]
shooting = df[df['Label'].str.contains('shooting')]
walking = df[df['Label'].str.contains('walking')]


# In[7]:


#Importing and Label Encoding 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('df_activity.csv', index_col = 0)
label = LabelEncoder()
df['Label'] = label.fit_transform(df['Label'].values)

df_new = df.drop(['Subjects'], axis = 1)

df_s1 = df.loc[df['Subjects'] == 'Subject1']
d1_s1 = df_s1.drop(['Subjects'], axis = 1)

df_s2 = df.loc[df['Subjects'] == 'Subject2']
d2_s2 = df_s2.drop(['Subjects'], axis = 1)

df_cp = df_new.loc[df_new["Label"] != 1]

window_length = 100
overlap = 60


# In[8]:


#Sliding Window
def sliding_window_samples(data, samples_per_window, overlap_ratio):
    """
    Return a sliding window measured in number of samples over a data array.

    :param data: input array, can be numpy or pandas dataframe
    :param samples_per_window: window length as number of samples per window
    :param overlap_ratio: overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    win_len = samples_per_window
    if overlap_ratio is not None:
        overlapping_elements = int((overlap_ratio / 100) * (win_len))
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements
    try:
        result_windows = np.array(windows)
        result_indices = np.array(indices)
    except:
        result_windows = np.empty(shape=(len(windows), win_len, data.shape[1]), dtype=object)
        result_indices = np.array(indices)
        for i in range(0, len(windows)):
            result_windows[i] = windows[i]
            result_indices[i] = indices[i]

    return result_windows, result_indices


# In[9]:


#Reshaping the features and target for giving input to the model
#It takes the most common values in each window for the target after appling sliding window and 
#also returns the reshaped array. Parameters assumed f=features and t=target
def reshape(f,t):
    axis = 1
    x_new = np.reshape(f, (f.shape[0], -1))
    y, indices = np.unique(t, return_inverse=True)
    y_new = y[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(t.shape),None, np.max(indices) + 1), axis=axis)]
    return np.asarray(x_new), np.asarray(y_new)


# **KNN Model**

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

def model_KNN(features, target, score):
    knn = KNeighborsClassifier(n_neighbors = 3,  p = 2)
    kfold = KFold(n_splits = 10, random_state=42, shuffle=True)
    scores = cross_val_score(knn, features, target, cv = kfold, scoring = score, n_jobs = -1)
    return scores.mean()


# In[11]:


#Normal Cross Validation for KNN
df_1, df_2 = sliding_window_samples(df_new, window_length, overlap)
data = np.array(df_1, dtype = float)
X_data = data[:, : ,0:12]
y_data = data[:, : , 12]
X_n, y_n = reshape(X_data, y_data)
accuracy_n = model_KNN(X_n, y_n, 'accuracy')
precision_n = model_KNN(X_n, y_n, 'precision_weighted')
recall_n =  model_KNN(X_n, y_n, 'recall_weighted')
f1_n = model_KNN(X_n, y_n, 'f1_weighted')

#Per Participant Cross Validation for KNN
#Subject 1
df_sub1, df_subj_1 = sliding_window_samples(d1_s1, window_length, overlap)
data_s1 = np.array(df_sub1, dtype = float)
X_s1 = data_s1[:, : ,0:12]
y_s1 = data_s1[:, : , 12]
X_sub1, y_sub1 = reshape(X_s1, y_s1)
accuracy_s1 = model_KNN(X_sub1, y_sub1, 'accuracy')
precision_s1 = model_KNN(X_sub1, y_sub1, 'precision_weighted')
recall_s1 = model_KNN(X_sub1, y_sub1, 'recall_weighted')
f1_s1 = model_KNN(X_sub1, y_sub1, 'f1_weighted')

#Subject 2
df_sub2, df_subj_2 = sliding_window_samples(d2_s2, window_length, overlap)
data_s2 = np.array(df_sub2, dtype = float)
X_s2 = data_s2[:, : ,0:12]
y_s2 = data_s2[:, : , 12]
X_sub2, y_sub2 = reshape(X_s2, y_s2)
accuracy_s2 = model_KNN(X_sub2, y_sub2, 'accuracy')
precision_s2 = model_KNN(X_sub2, y_sub2, 'precision_weighted')
recall_s2 = model_KNN(X_sub2, y_sub2, 'recall_weighted')
f1_s2 = model_KNN(X_sub2, y_sub2, 'f1_weighted')


# In[ ]:





# In[ ]:





# **LSTM model**

# In[12]:


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
from pandas import DataFrame

def model_LSTM(X,y,k):
    scores=[]
    f1=[]
    matrix=[]
    split = 1
    
    """Different types of Kfold for splitting the data
    :rkfold = RepeatedKFold(n_splits=k, random_state=42)
    :kfold = KFold(n_splits=k,random_state=42, shuffle=True)"""
    
    skf = StratifiedKFold(n_splits=k,random_state=None, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        print('Fold number : ------------- : ', split)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Model architecture            
        model = Sequential()
        model.add(LSTM(units=26,input_shape=[X_train.shape[1], X_train.shape[2]], activation = 'relu', return_sequences = True))
        model.add(Dropout(rate=0.1))
        model.add(Flatten())
        model.add(Dense(units=130, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=850, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=6, activation='softmax'))
        
        #Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Fit data to model
        model.fit(X_train,y_train,epochs=4, validation_data=(X_test,y_test),verbose=1)
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        
        #Genarate Metrics
        scores.append(model.evaluate(X_test,y_test))
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        matrix.append(confusion_matrix(y_test, y_pred))
        split+=1 
         
    return np.array(scores), np.array(f1), np.array(matrix)


# In[13]:


#Normal Validation
df_1, df_2 = sliding_window_samples(df_new, window_length, overlap)
data = np.array(df_1, dtype = float)

X_n = data[:, : ,0:12]
y_n = data[:, : , 12]
X_dummy, y_n = reshape(X_n, y_n)
fold_n=10
scores_n, f1_scores_n, matrix_n = model_LSTM(X_n,y_n,fold_n)
acc_scores_n = scores_n[:,-1]
c_matrix_n = matrix_n[-1]


# In[14]:


#Per Participant Cross Validation
#Subject 1
df_sub1, df_subj_1 = sliding_window_samples(d1_s1, window_length, overlap)
data_s1 = np.array(df_sub1, dtype = float)

X_s1 = data_s1[:, : ,0:12]
y_s1 = data_s1[:, : , 12]
X_dummy, y_s1 = reshape(X_s1, y_s1)
fold_s1=4
scores_s1, f1_scores_s1, matrix_s1 = model_LSTM(X_s1,y_s1,fold_s1)
acc_scores_s1 = scores_s1[:,-1]
c_matrix_s1 = matrix_s1[-1]


# In[15]:


#Subject 2
df_sub2, df_subj_2 = sliding_window_samples(d2_s2, window_length, overlap)
data_s2 = np.array(df_sub2, dtype = float)

X_s2 = data_s2[:, : ,0:12]
y_s2 = data_s2[:, : , 12]
X_dummy, y_s2 = reshape(X_s2, y_s2)
fold_s2=8
scores_s2, f1_scores_s2, matrix_s2  = model_LSTM(X_s2,y_s2,fold_s2)
acc_scores_s2 = scores_s2[:,-1]
c_matrix_s2 = matrix_s2[-1]


# In[16]:


#Cross Participants Cross Validation
df_cp_sw, df_sw_index = sliding_window_samples(df_cp, window_length, overlap)
data_cp = np.array(df_cp_sw, dtype = float)

X_cp = data_cp[:, : ,0:12]
y_cp = data_cp[:, : , 12]
X_dummy, y_cp = reshape(X_cp, y_cp)
fold_cp=2
scores_cp, f1_scores_cp, matrix_cp = model_LSTM(X_cp,y_cp,fold_cp)
acc_scores_cp = scores_cp[:,-1]
c_matrix_cp = matrix_cp[-1]


# In[17]:


print('Normal Cross Validation, Model : KNN')
print('Mean_accuracy: %.3f' % accuracy_n)
print('Mean_precision_score: %.3f' % precision_n)
print('Mean_recall_score: %.3f' % recall_n)
print('Mean_F1_score: %.3f' % f1_n)

print('\nSUB 1 Cross Validation, Model : KNN')
print('Mean_accuracy_s1: %.3f' % accuracy_s1)
print('Mean_precision_score_s1: %.3f' % precision_s1)
print('Mean_recall_score_s1: %.3f' % recall_s1)
print('Mean_F1_score_s1: %.3f' % f1_s1)

print('\nSUB 2 Cross Validation, Model : KNN')
print('Mean_accuracy_s2: %.3f' % accuracy_s2)
print('Mean_precision_score_s2: %.3f' % precision_s2)
print('Mean_recall_score_s2: %.3f' % recall_s2)
print('Mean_F1_score_s2: %.3f' % f1_s2)

print('\nNormal Cross Validation, Model : LSTM')
print('Accuracy : %0.4f' %acc_scores_n.mean())
print('F1_score : %0.4f' %f1_scores_n.mean())
print('Confusion_matrix : Null, dribbling, layup, running, shooting, walking\n', c_matrix_n)

print('\nSUB 1 Cross Validation, Model : LSTM')
print('Accuracy : %0.4f' %acc_scores_s1.mean())
print('F1_score : %0.4f' %f1_scores_s1.mean())
print('Confusion_matrix : Null, layup, running, shooting, walking \n', c_matrix_s1)

print('\nSUB 2 Cross Validation, Model : LSTM')
print('Accuracy : %0.4f' %acc_scores_s2.mean())
print('F1_score : %0.4f' %f1_scores_s2.mean())
print('Confusion_matrix : Null, dribbling, layup, running, shooting, walking \n', c_matrix_s2)

print('\nCross Participants Cross Validation, Model : LSTM')
print('Accuracy : %0.4f' %acc_scores_cp.mean())
print('F1_score : %0.4f' %f1_scores_cp.mean())
print('Confusion_matrix : Null, layup, running, shooting, walking \n', c_matrix_cp)


# In[ ]:




