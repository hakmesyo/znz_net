# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:23:14 2021

@author: DELL LAB
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import load_model
from PIL import Image, ImageOps
import pandas as pd
from distutils.dir_util import copy_tree
from random import shuffle
from random import seed
import shutil
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def test_tensorflow_gpu():
    return tf.test.is_built_with_cuda()

def test_gpu():
    return tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    
def show_samples(train_dataset):
    class_names = train_dataset.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Note: These layers are active only during training, when you call model.fit. They are inactive when the model is used in inference mode in model.evaulate or model.fit.
def show_data_aug_images(train_dataset):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ]) 
    for image, _ in train_dataset.take(1):
      plt.figure(figsize=(10, 10))
      first_image = image[0]
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
    return data_augmentation

def train_model(
        train_dir,
        valid_dir,
        test_dir,
        BATCH_SIZE=32,
        IMG_SIZE = (224,224),
        IMG_CHANNEL = 3,
        initial_epochs = 10,
        fine_tune_epochs = 10,
        model_path=None,
        isBinaryClassification=False,
        isAccPlotVisible=False,
        isSampleImagesVisible=False,
        isDataAugmentation=False,
        isDataAugImageVisible=False
        
                ):

    IMG_SHAPE = IMG_SIZE + (IMG_CHANNEL,)
    
    total_epochs=initial_epochs+fine_tune_epochs
    dirs=os.listdir(train_dir)
    res=range(len(dirs))
    OUTPUTS=len(dirs)
    restrain=[]
    resvalid=[]
    restest=[]

    for inx,files in enumerate(os.listdir(train_dir)):
        for file in os.listdir(os.path.join(train_dir,files)):
            restrain.append([os.path.join(files,file),files.lower(),inx])
    
    for inx,files in enumerate(os.listdir(valid_dir)):
        for file in os.listdir(os.path.join(valid_dir,files)):
            resvalid.append([os.path.join(files,file),files.lower(),inx])
            
    for inx,files in enumerate(os.listdir(test_dir)):
        for file in os.listdir(os.path.join(test_dir,files)):
            restest.append([os.path.join(files,file),files.lower(),inx])
    
    datatrain = pd.DataFrame(restrain, columns=['FileName', 'ClassName', 'ClassNumber'])
    datatrain.dropna(inplace=True) 
    
    datavalid = pd.DataFrame(resvalid, columns=['FileName', 'ClassName', 'ClassNumber'])
    datavalid.dropna(inplace=True) 
    
    datatest = pd.DataFrame(restest, columns=['FileName', 'ClassName', 'ClassNumber'])
    datatest.dropna(inplace=True)     

    dftrain=datatrain[datatrain['ClassNumber'].isin(res)]
    dfvalid=datavalid[datavalid['ClassNumber'].isin(res)]
    dftest=datatest[datatest['ClassNumber'].isin(res)]

    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input            
    )
    
    train_dataset=datagen_train.flow_from_dataframe(
        dataframe=dftrain,
        directory=train_dir,
        x_col="FileName",
        y_col="ClassName",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode="sparse",
        target_size=IMG_SIZE
    )
    
    valid_dataset=datagen_train.flow_from_dataframe(
        dataframe=dfvalid,
        directory=valid_dir,
        x_col="FileName",
        y_col="ClassName",
        batch_size=BATCH_SIZE,
        seed=42,
        # shuffle=True,
        class_mode="sparse",
        target_size=IMG_SIZE
    )
    
    test_dataset=datagen_train.flow_from_dataframe(
        dataframe=dftest,
        directory=test_dir,
        x_col="FileName",
        y_col="ClassName",
        batch_size=BATCH_SIZE,
        seed=42,
        # shuffle=True,
        class_mode="sparse",
        target_size=IMG_SIZE
    )
    
    
    if isSampleImagesVisible:
        show_samples(train_dataset)
    
    
    # Make the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # This feature extractor converts each 224x224x3 image into a 5x5x1280 block of features. Let's see what it does to an example batch of images:
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    
    base_model.trainable = False
    '''
    Important note about BatchNormalization layers
    Many models contain tf.keras.layers.BatchNormalization layers. 
    This layer is a special case and precautions should be taken in the context of 
    fine-tuning, as shown later in this tutorial.
    
    When you set layer.trainable = False, the BatchNormalization layer will run 
    in inference mode, and will not update its mean and variance statistics.
    
    When you unfreeze a model that contains BatchNormalization layers in order 
    to do fine-tuning, you should keep the BatchNormalization layers in inference 
    mode by passing training = False when calling the base model. Otherwise, 
    the updates applied to the non-trainable weights will destroy what the model has learned.
    '''
    # base_model.summary()
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)
    
    prediction_layer = tf.keras.layers.Dense(OUTPUTS,activation='sigmoid')
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)
    
    
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    if isDataAugmentation:
        # data_augmentation = tf.keras.Sequential([
        #   tf.keras.layers.RandomFlip('horizontal'),
        #   tf.keras.layers.RandomRotation(0.2),
        # ])
        # tf.keras.preprocessing.image.ImageDataGenerator(
        #     featurewise_center=False, samplewise_center=False,
        #     featurewise_std_normalization=False, samplewise_std_normalization=False,
        #     zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
        #     height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
        #     channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
        #     horizontal_flip=False, vertical_flip=False, rescale=None,
        #     preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
        # )
        data_augmentation=tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=30, horizontal_flip=True)
        if isDataAugImageVisible:
            for image, _ in train_dataset.take(1):
              plt.figure(figsize=(10, 10))
              first_image = image[0]
              for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')

        
        x = data_augmentation(inputs)
        # x = preprocess_input(x)
    else:
        # x = preprocess_input(inputs)
        pass
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    base_learning_rate = 0.0001
    if isBinaryClassification:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'])
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),              
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),              
            metrics=['accuracy'])
        
    model.summary()
    
    
    loss0, accuracy0 = model.evaluate(valid_dataset)
    
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=valid_dataset)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False
    
    if isBinaryClassification:
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
            metrics=['accuracy'])
    else:
        model.compile(
            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
            metrics=['accuracy'])
        
    model.summary()
    
    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=valid_dataset)
    
    
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']
    
    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']
    
    if isAccPlotVisible:
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.1, 1])
        plt.plot([initial_epochs-1,initial_epochs-1],
                  plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 5.0])
        plt.plot([initial_epochs-1,initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
    
    
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy*100)
    
    if model_path!=None:
        model.save(model_path)
    
    return model


# Örnek kullanımı
# base_path=r'C:\python_data\dataset\gatenet\datasets\pollens_224\fat_model'
# train_dir = os.path.join(base_path, 'train')
# valid_dir = os.path.join(base_path, 'valid')
# test_dir = os.path.join(base_path, 'test')
# model_path = r'C:\python_data\dataset\gatenet\datasets\pollens_224\models\deneme.h5'


# train_model(
#         train_dir, 
#         valid_dir, 
#         test_dir, 
#         isAccPlotVisible=True,
#         isSampleImagesVisible=True,
#         initial_epochs=10,
#         fine_tune_epochs=10,
#         model_path=model_path
#     )

def train_zanim_nezanim_model(
        base_path,
        train_path,
        valid_path,
        test_path,
        temp_path,
        model_path,
        range_of,
        BATCH_SIZE=32,
        IMG_SIZE = (224,224),
        IMG_CHANNEL = 3,
        initial_epoch=10,
        final_epoch=10,
        train_sample_size=20,
        valid_sample_size=2,
        test_sample_size=2
    ):
    
    mx=train_sample_size
    train_dir=train_path
    valid_dir=valid_path
    test_dir=test_path
    dirs=os.listdir(train_dir)
    for i in range_of: 
        isExist = os.path.exists(temp_path)
        if isExist:
            shutil.rmtree(temp_path)
        os.makedirs(temp_path)
        os.makedirs(temp_path+'/train')
        os.makedirs(temp_path+'/valid')
        os.makedirs(temp_path+'/test')
        
        os.makedirs(temp_path+'/train/'+dirs[i])
        copy_tree(train_dir+'/'+dirs[i], temp_path+'/train/'+dirs[i])
        os.makedirs(temp_path+'/train/znz')
        
        os.makedirs(temp_path+'/valid/'+dirs[i])
        copy_tree(valid_dir+'/'+dirs[i], temp_path+'/valid/'+dirs[i])
        os.makedirs(temp_path+'/valid/znz')
        
        os.makedirs(temp_path+'/test/'+dirs[i])
        copy_tree(test_dir+'/'+dirs[i], temp_path+'/test/'+dirs[i])
        os.makedirs(temp_path+'/test/znz')
        
        
        for j in range(len(dirs)):
            if j==i:
                continue
            print(dirs[j])
            img_paths=os.listdir(os.path.join(train_dir,dirs[j]))
            seed(1234)
            shuffle(img_paths)
            if len(img_paths)<mx:
                mx=len(img_paths)
            for k in range(mx):
                shutil.copy(train_dir+'/'+dirs[j]+'/'+img_paths[k], temp_path+'/train/znz/'+'/'+img_paths[k])
            
            img_paths=os.listdir(os.path.join(valid_dir,dirs[j]))
            seed(1234)
            shuffle(img_paths)
            for k in range(valid_sample_size):
                shutil.copy(valid_dir+'/'+dirs[j]+'/'+img_paths[k], temp_path+'/valid/znz/'+'/'+img_paths[k])
            
            img_paths=os.listdir(os.path.join(test_dir,dirs[j]))
            seed(1234)
            shuffle(img_paths)
            for k in range(test_sample_size):
                shutil.copy(test_dir+'/'+dirs[j]+'/'+img_paths[k], temp_path+'/test/znz/'+'/'+img_paths[k])
            
        seed()
        train_model(
                temp_path+'/train', 
                temp_path+'/valid', 
                temp_path+'/test', 
                BATCH_SIZE,
                IMG_SIZE,
                IMG_CHANNEL,
                # isAccPlotVisible=True,
                # isSampleImagesVisible=True,
                initial_epochs=initial_epoch,
                fine_tune_epochs=final_epoch,
                model_path=model_path+'/'+str(i)+'.h5'
            )



def test_model_teachable_keras(model,base_path,images,width=224,height=224,channel=3):
    pred=[]
    for i in range(len(images)):
        data = np.ndarray(shape=(1, width, height, channel), dtype=np.float32)
        image = Image.open(os.path.join(base_path,images[i]))
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        print(i,'=',images[i])
        pred.append([np.argmax(prediction),prediction])
    return pred


def predictImage(model,path,width=224,height=224,channel=3):
    data = np.ndarray(shape=(1, width, height, channel), dtype=np.float32)
    image = Image.open(path)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)#(image/127.5) - 1
    data[0]=image
    # plt.imshow(np.squeeze(data))
    prediction = model.predict(data)
    # print(prediction.argmax())
    return prediction


def test_model_whole(model,test_dir,width=224,height=224,channel=3):
    dirs=os.listdir(test_dir)
    res=range(len(dirs))
    hit=0
    err=0
    acc=0 
    accs=list(np.zeros(len(dirs)))
    hits=list(np.zeros(len(dirs)))
    errs=list(np.zeros(len(dirs)))  
    for i in res:
        file_list=os.listdir(test_dir+'/'+dirs[i])
        for k in range(len(file_list)):
            path=os.path.join(test_dir,dirs[i],file_list[k])
            p=predictImage(model, path,width,height,channel)
            index=np.argmax(p)
            if index==i:
                hit+=1
                hits[i]+=1
            else:
                err+=1
                errs[i]+=1
            acc=round(hit/(hit+err)*100,2)
            print(' acc=',acc,' hit:',hit,' err:',err)
            accs[i]=round(hits[i]/(hits[i]+errs[i])*100,2)
            pass
    return acc,hit,err,accs,hits,errs

def test_model_specific_for_fat_model(model,class_index,test_dir,width=224,height=224,channel=3):
    hit=0
    err=0
    acc=0 
    recall=0
    file_list=os.listdir(test_dir)
    preds=[]
    for k in range(len(file_list)):
        path=os.path.join(test_dir,file_list[k])
        p=predictImage(model, path,width,height,channel)
        preds.append(p)
        index=np.argmax(p)
        if index==class_index:
            hit+=1
        else:
            err+=1
        acc=round(hit/(hit+err)*100,4)
        recall=round(hit/(hit+err),4)
        print(' acc=',acc,' hit:',hit,' err:',err,' recall:',recall)
        pass           
    
    return acc,hit,err,preds,recall

def test_model_specific(model,class_index,test_dir,width=224,height=224,channel=3):
    hit=0
    err=0
    acc=0 
    recall=0
    file_list=os.listdir(test_dir)
    preds=[]
    for k in range(len(file_list)):
        path=os.path.join(test_dir,file_list[k])
        p=predictImage(model, path,width,height,channel)
        preds.append(p)
        index=np.argmax(p)
        if index==0:
            hit+=1
        else:
            err+=1
        acc=round(hit/(hit+err)*100,4)
        recall=round(hit/(hit+err),4)
        print(' acc=',acc,' hit:',hit,' err:',err,' recall:',recall)
        pass           
    
    return acc,hit,err,preds,recall

def load_znz_models(model_base_path,range_of):
    models=[]
    file_list=os.listdir(model_base_path)
    for i in range_of:
        models.append(load_model(model_base_path+'/'+str(i)+".h5"))
        print(i,'.model was loaded!')
    return models

def generate_test_set_from_dir_dirs_files(test_dir,BATCH_SIZE=16,IMG_SIZE=(224,224)):
    dirs=os.listdir(test_dir)
    res=range(len(dirs))
    restest=[]
    for inx,files in enumerate(os.listdir(test_dir)):
        for file in os.listdir(os.path.join(test_dir,files)):
            restest.append([os.path.join(files,file),files.lower(),inx])
    
    datatest = pd.DataFrame(restest, columns=['FileName', 'ClassName', 'ClassNumber'])
    datatest.dropna(inplace=True)     

    dftest=datatest[datatest['ClassNumber'].isin(res)]

    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input            
    )
    test_dataset=datagen_train.flow_from_dataframe(
        dataframe=dftest,
        directory=test_dir,
        x_col="FileName",
        y_col="ClassName",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=False,
        class_mode="sparse",
        target_size=IMG_SIZE
    )   
    return test_dataset


def generate_test_from_dir_files(test_dir,BATCH_SIZE=32,IMG_SIZE=(224,224)):
    restest=[]
    
    for files in os.listdir(test_dir):
        restest.append([test_dir,files.lower(),str(0)])
    
    datatest = pd.DataFrame(restest, columns=['FileName', 'ClassName', 'ClassNumber'])
    datatest.dropna(inplace=True) 
    
    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input            
    )
    
    test_dataset=datagen_train.flow_from_dataframe(
        dataframe=datatest,
        directory=test_dir,
        x_col="ClassName",
        y_col="ClassNumber",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=False,
        class_mode="sparse",
        target_size=IMG_SIZE
    )   
    return test_dataset

def get_performance_metrics(model, test_dataset,dirs=None):
    # loss, accuracy = model.evaluate(test_dataset, verbose=0, steps=1)
    # print("Test Accuracy\t{0:.6f}".format(accuracy))
    # test_dataset.reset()
    STEP_SIZE_TEST=test_dataset.n//test_dataset.batch_size
    Y_test_pred = model.predict(test_dataset, steps=STEP_SIZE_TEST+1,verbose=1)
    acscore = accuracy_score(np.array(test_dataset.labels),np.argmax(Y_test_pred, axis=1))
    print("Prediction accuracy=\t"+str(acscore))
    
    pcs=precision_score(np.array(test_dataset.labels),np.argmax(Y_test_pred, axis=1),average='micro')  #Doğruluk ,average='binary'
    print("Precision score=\t"+str(pcs))
    
    recs=recall_score(np.array(test_dataset.labels),np.argmax(Y_test_pred, axis=1),average='micro')  #Doğruluk ,average='binary'
    print("Recall score=\t"+str(recs))
   
    f1=f1_score(np.array(test_dataset.labels),np.argmax(Y_test_pred, axis=1),average='micro')  #Doğruluk      
    print("F1 score=\t"+str(f1))
    
    cm=confusion_matrix(np.array(test_dataset.labels),np.argmax(Y_test_pred, axis=1)) #Hata matrisi
    print(cm)
    
    classification_metrics=0
    if dirs!=None:
        y_true=test_dataset.labels
        y_pred=Y_test_pred
        y_pred = np.argmax(y_pred,axis=1)
        classification_metrics=metrics.classification_report(y_true, y_pred, target_names=dirs)
        print(classification_metrics)
    
    return acscore,pcs,recs,f1,cm,classification_metrics

    
def test_batch_for_fat_model(model,test_dir):
    dirs=os.listdir(test_dir)
    test_dataset=generate_test_set_from_dir_dirs_files(test_dir)    
    return get_performance_metrics(model, test_dataset,dirs)
    
def test_batch_for_znz_model(model,test_dir):
    test_dataset=generate_test_from_dir_files(test_dir)    
    return get_performance_metrics(model, test_dataset,dirs=None)

# def test_batch():
#     loss, accuracy = model.evaluate(test_generator, verbose=0, steps=STEP_SIZE_TEST)
#     print("Test Accuracy\t{0:.6f}".format(accuracy))
#     test_generator.reset()
#     Y_test_pred = model.predict(test_generator, verbose=0, steps=STEP_SIZE_TEST+1)
    
def plot_confusion_matrix(model, x, y, plot_title = ''):
    y_pred = model.predict(x)                            # get predictions on x using model
    predicted_categories = tf.argmax(y_pred, axis=1)     # get index of predicted category
    true_categories = tf.argmax(y, axis=1)               # get index of true category
    # create confusion matrix using sklearn
    cm = confusion_matrix(true_categories, predicted_categories)
    # create DataFrame from the confusion matrix. We retrieve labels from LabelEncoder.
    df_cm = pd.DataFrame(cm, index = le.classes_ ,  columns = le.classes_)
    # divide each row to its sum in the DataFrame to get normalized output
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    
    plt.figure(figsize = (15,12))
    plt.title(plot_title)
    sns.heatmap(df_cm, annot=True)
