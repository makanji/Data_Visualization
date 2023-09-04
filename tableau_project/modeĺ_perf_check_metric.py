#%%
#library importation
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow.keras.layers as layers
import keras.preprocessing.image as IMG
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import time
import timeit
import copy
import random
import sys
import math
import statistics
import skimage
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
from skimage.feature import blob_dog,blob_log,blob_doh
from sklearn.neighbors import DistanceMetric
from math import sqrt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn import cluster as cl
from sklearn.cluster import KMeans

#parameter definition
original_dim = (2160, 3840)
Encoder_Input_Dim = (800, 1200, 1)
Encoder_Ouptu_Dim = (50, 75, 64)
color_channel = 1
classifier_Output_Dim = ( 400, 600, 4)
input_shape = (50, 75, 64)
width= classifier_Output_Dim[1]
height= classifier_Output_Dim[0] # this will make bloob detection in thesame dimenrion with our dim for metric cal


#model definitions and dependance
chosen_weight = "Classifier_600_Epoch_151.h5"
saved_weight = "/Home/makanji/Master_Thesis/turckey_classifier/" #customised loss
#saved_weight = "/Home/makanji/Master_Thesis/Turkey_classifier_loss_BCE/" #BCE loss
chosen_weight_path = saved_weight + chosen_weight
from tensorflow.keras.models import load_model
loadedEncoder = load_model("Turkey_Encoder.h5", compile=False)

loadModel = load_model(chosen_weight_path, compile=False)
frame_dir = "/Home/makanji/Master_Thesis/Datasets/Turkey5_Grp/Turkey5_Grp_4/"
#anno_path = testf_annot = "/Home/makanji/Master_Thesis/Datasets/Turkey5_Grp/" + "Turkey5_Grp_4Annotation.csv"
# using test annotatation
anno_path = testf_annot = "/Home/makanji/Master_Thesis/Datasets/Turkey5_Grp/" + "TestsetsAnnotation.csv"

# Load the annotations
##################################################################################
pd.set_option('display.max_columns', None)
metadata = pd.read_csv(anno_path)
metadata.head(10)
# Load the frames, make predictions and visualize the results.
frameNames = os.listdir(frame_dir)
metadata = metadata.drop_duplicates()
meta_img_names = metadata['fileName'].unique()
len(frameNames)

#%%
# importation of models to check
#aved_weight = "/Home/makanji/Master_Thesis/turckey_classifier/" #customised loss
#saved_weight = "/Home/makanji/Master_Thesis/Turkey_classifier_loss_BCE/" #BCE loss
saved_weight = "/Home/makanji/Master_Thesis/mode_saved_epoch/model_900/"
model_perf = []
final_score = []
counter = 0
for model_index in os.listdir(saved_weight):
    current_name = saved_weight + model_index
    TP_total = 0
    FP_total = 0
    FN_total = 0
    Trky_total = 0
    orig_frames = []
    frame_preds = []

    #loadmodel
    print(f"Model index {counter} with name {current_name.split('/')[-1]} is been uploaded")
    loadModel = load_model(current_name, compile=False)

    # importation of image into the session
    test_file_nr = (len(frameNames))  # this for tracking the metric additions through out the loop
    for i in range(len(frameNames)):
        print(f"Frame with index number: {i} is been processed ")
        meta_sub = metadata.groupby('fileName')
        sub_sample = image_annotations = meta_sub.get_group(meta_img_names[i])
        imported_image = cv2.imread(frame_dir + sub_sample.fileName.unique()[0], 0) # loading image now with the saved name
        orig_frames.append(imported_image)
        orig_dim_Y = imported_image.shape[0]    # dimensional check before img resize
        orig_dim_X = imported_image.shape[1]
        turkey_point = []   # getting every point for turkey on each frames # using dimension 400, 600
        for j in range(len(sub_sample)):
            turkey_ptX = int(image_annotations['p2'].iloc[j].split(",")[0])
            turkey_ptY = int( image_annotations['p2'].iloc[j].split(",")[1])
            # rescale the par_2 to the CNN target dimension
            y_factor = orig_dim_Y / height
            x_factor = orig_dim_X / width

            turkey_ptY = int(turkey_ptY / y_factor)
            turkey_ptX = int(turkey_ptX / x_factor)
            turkey_point.append([turkey_ptY, turkey_ptX])

        ##########################################################################
        #making prediction and visualizing it
        curr_frame = copy.deepcopy(orig_frames[i])
        curr_frame =curr_frame/255
        present_extracted_img = cv2.resize(curr_frame, (width*2, height*2))
        #present_extracted_img = x_test[i, :, :]
        img_for_cv = copy.deepcopy(present_extracted_img)
        img_for_encoder = (copy.deepcopy(present_extracted_img)).reshape(1, Encoder_Input_Dim[0], Encoder_Input_Dim[1],
                                                                         Encoder_Input_Dim[2])
        present_encoded_img = loadedEncoder.predict(img_for_encoder)
        encoded_test = loadModel.predict(present_encoded_img)
        extract_hd_img = encoded_test[0, :, :, 1]
        frame_preds.append(extract_hd_img)
        ################################################################################
        #using blob detection to extract the model predictions

        blobs = blob_log(frame_preds[i], max_sigma=450, min_sigma=13, threshold=0.3, overlap=0.01)
        centers = blobs[:, 0:2]
        # centers = centers * 2 # rescaling by the factors
        shoulder_coors_frame = centers
        for pred_i in range(len(centers)):
            position_x = int(centers[pred_i][1])
            postition_y = int(centers[pred_i][0])
        # to switch to visualise the images with pred, orig position and radius just use previous image instead of newly copied
        # saving the images might be better than just the single image earlier

        # performance metrics on each frames
        circle_center = turkey_point
        blob_center = centers
        radius = 10

        tp = []
        fp = []
        fn = []

        for img_i in range(len(circle_center)):
            # print(i)
            x_center, y_center = circle_center[img_i][0], circle_center[img_i][1]
            # print(x_center)
            # print(y_center)

            tp_tmp = []
            fp_tmp = []
            empty_list = []

            for j in range(len(blob_center)):
                b_center_x, b_center_y = blob_center[j][0], blob_center[j][1]

                # check if blob center is in circle
                if ((b_center_x - x_center) ** 2 + (b_center_y - y_center) ** 2) <= radius ** 2:
                    # print(" blob is in center")

                    if len(tp_tmp) == 0:
                        tp_tmp.append(1)
                        tp.append(1)
                    else:
                        fp_tmp.append(1)

                elif ((b_center_x - x_center) ** 2 + (b_center_y - y_center) ** 2) > radius ** 2:
                    # print("Blob is not in the circle")
                    empty_list.append(1)

                if len(empty_list) == len(blob_center):
                    fn.append(1)

                nr_fp_tmp = len(fp_tmp)
                for h in range(nr_fp_tmp):
                    fp.append(1)

        # new loop for finding the blobs that are outside of any circle
        for j in range(len(blob_center)):
            b_center_x, b_center_y = blob_center[j][0], blob_center[j][1]

            in_a_circle_list = []

            # ask if the current blob is in any of the circles
            for img_i in range(len(circle_center)):
                x_center, y_center = circle_center[img_i][0], circle_center[img_i][1]

                if ((b_center_x - x_center) ** 2 + (b_center_y - y_center) ** 2) < radius ** 2:
                    in_a_circle_list.append(1)

            if len(in_a_circle_list) == 0:
                fp.append(1)


        TP_total += len(tp)
        FP_total += len(fp)
        FN_total += len(fn)
        Trky_total = Trky_total + len(turkey_point) # true coordinate points


        # Sensitivity (= Recall)
        sensitivity = round(TP_total / Trky_total, ndigits = 3)

        #Precision
        precision =  round( TP_total / (TP_total + FP_total) , ndigits = 3)

        # False discovery rate
        FDR =   round(FP_total / (TP_total + FP_total) , ndigits = 3)

        # F1-score
        F1 =  round( (2 * TP_total) / ( 2 * TP_total + FP_total + FN_total ) , ndigits = 3)

        if i == test_file_nr - 1:
            print("################################ this the end result ################################ ")
            final_score.append([current_name.split('/')[-1], sensitivity, precision, FDR, F1])
            print("The values for sensitivity, precision, false-discovery rate and F1-score are: \n")
            print("     Sensitivity: %s\n     Precision: %s\n     FDR: %s\n     F1-score: %s\n" % (sensitivity, precision, FDR, F1))


    counter += 1
#%%

list_explore = copy.deepcopy(final_score)
len(list_explore)

#type(list_explore)
df_metric_score = pd.DataFrame(list_explore, columns = ['Name_of_model', 'Sensitivity', 'Precision' , 'FDR', 'F1_score'])
#df_metric_score.shape
file_name = "/Home/makanji/Master_Thesis/Datasets/Metric_perf/model_900_img.xlsx"
df_metric_score.to_csv(file_name)
df_metric_score.to_excel(file_name)

final_score[-1]

#%%
#writing csv

filename = "/Home/makanji/Master_Thesis/Datasets/Metric_perf/model_900_img.csv"
import csv

# field names
fields = ['Name_of_model', 'Sensitivity', 'Precision' , 'FDR', 'F1_score']

# data rows of csv file
rows = final_score

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)

