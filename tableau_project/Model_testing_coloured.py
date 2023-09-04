#%%
# library importations
#libraries importations
from skimage.feature import blob_log
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import cv2
from tensorflow.keras.models import load_model

# parameters definitions
Encoder_Input_Dim = (800, 1200, 1)
Encoder_Ouptu_Dim = (50, 75, 64)
color_channel = 1
classifier_Output_Dim = ( 400, 600, 4)
input_shape = (50, 75, 64)

# loading up classifier model and encoder model
saved_weight = "/Home/makanji/Master_Thesis/turckey_classifier/"
chosen_weight_path = "/Home/makanji/Master_Thesis/mode_saved_epoch/model_all_imgs/Classifier_1717_Epoch_165.h5"
path_extracted_pred = "/Home/makanji/Master_Thesis/Datasets/Created_test_color/" # folder in use for predicted frames
file_ext_path = "/Home/makanji/Master_Thesis/Datasets/model_deploy_color/" #folder in use
sample_video = "/Home/makanji/Master_Thesis/Datasets/video_sample/20_01_R_20220712093000_0_30.avi"
loadedEncoder = load_model("Turkey_Encoder.h5", compile=False)
loadModel = load_model(chosen_weight_path, compile=False)

#binary image with feeder and drinker
feeder = [[138, 234]] #radius = 36
feeder_radius = 36
drinker = [[337, 112]] #radius = 21
drinker_radius = 21
bin_frame = np.zeros(shape=(classifier_Output_Dim[0], classifier_Output_Dim[1], 3), dtype=float) # color channels
binframe_use = copy.deepcopy(bin_frame)
binary_img= cv2.circle(binframe_use, (337, 112), radius= 21, color=(1, 1, 1), thickness=2) #drinker
binary_img = cv2.circle(binframe_use, (138, 234), radius= 36, color=(1,1,1), thickness=2) #feeder
plt.imshow(binary_img)

#created folders
joined_images = []

#%%
# function definitions
# frame extractions
class VideoFrames:
    """Class for extracting frames from a video."""
    def __init__(self, file, rate):
        self.file = file
        self.rate = rate
    def extract_frames(self, out='./images/'):
        # Read in video stream
        cap = cv2.VideoCapture(self.file)
        success, frame = cap.read()
        i = 1
        # Get new frame if there are any left
        while success:
            frame_id = cap.get(1)
            # Only get each frame modulo rate and ignore others
            if self.rate == 1 or frame_id % self.rate == 1:
                filename = os.path.join(out, os.path.splitext(os.path.basename(self.file))[0] + '_frame' + str(i) + ".png")
                print(filename)
                cv2.imwrite(filename, frame)
            success, frame = cap.read()
            i = i + 1
        cap.release()

# frame processing
def frame_processing(img_file_path):
    """  This receives the raw image files process it into encoder input
    and final CV visualization"""
    im = cv2.imread(img_file_path, 0)
    im_vis = cv2.imread(img_file_path, 1)
    img_enc = cv2.resize(src=im, dsize=(Encoder_Input_Dim[1], Encoder_Input_Dim[0]), interpolation=cv2.INTER_AREA)
    img_cv = cv2.resize(src=im, dsize=(classifier_Output_Dim[1], classifier_Output_Dim[0]), interpolation=cv2.INTER_AREA)
    img_vis = cv2.resize(src=im_vis, dsize=(classifier_Output_Dim[1], classifier_Output_Dim[0]), interpolation=cv2.INTER_AREA)
    # rescaling all image
    return img_enc/255, img_cv/255, img_vis/255

# model training and predictions
def extract_blobs (single_file):
    """This takes in the images and return a bloob detected files """
    img = single_file  # scaling of the inputted image
    img = img.reshape(1, Encoder_Input_Dim[0], Encoder_Input_Dim[1], Encoder_Input_Dim[2])  # reshaping the image
    enc_img = loadedEncoder.predict(img)
    pred_img = loadModel.predict(enc_img)
    extrct_bd = pred_img[0, :, :, 1] # extracting one frame amidst of the channels
    extrct_hd = pred_img[0, :, :, 0]
    extrct_tl = pred_img[0, :, :, 2]
    # generating the blobs from the single file
    bd_blobs = blob_log(extrct_bd, max_sigma=450, min_sigma=13, threshold=0.3, overlap=0.01)
    bd_coordinate = bd_blobs[:, 0:2]
    hd_blob = blob_log(extrct_hd, max_sigma=450, min_sigma=13, threshold=0.3, overlap=0.01)
    hd_coordinate = hd_blob[:, 0:2]
    tl_blob = blob_log(extrct_tl, max_sigma=450, min_sigma=13, threshold=0.3, overlap=0.01)
    tl_coordinate = tl_blob[:, 0:2]
    return hd_coordinate, bd_coordinate, tl_coordinate

# making use of the blob predicted by the network
def draw_blob_wif_coor (given_blob, img_color_in, radius = 5, color= (1, 1, 1) , thickness=cv2.FILLED):
    """This takes in the extracted blobs parameters and produce an images with the necessary postions """
    #binary_frame = np.zeros(shape=(classifier_Output_Dim[0], classifier_Output_Dim[1]), dtype=float)
    image = copy.deepcopy(img_color_in)
    img_color = copy.deepcopy(img_color_in)
    bin_frame_use = copy.deepcopy(binary_img)
    for i in range(len(given_blob)):
        position_x = int(given_blob[i][1])
        postition_y = int(given_blob[i][0])
        pred_frame = cv2.circle(image, (position_x, postition_y), radius=radius, color=color, thickness=thickness)
        pred_color_frame = cv2.circle(img_color, (position_x, postition_y), radius=radius, color= color, thickness=thickness)
        binary_pred = cv2.circle(bin_frame_use , (position_x, postition_y), radius=radius, color=color, thickness=thickness)
    return pred_frame, pred_color_frame, binary_pred

#checking number of animals using the functional areas
def circle_count (par_1, par_2, radius):
    """this give the sum of points in a given circle"""
    count = []
    for img_i in range(len(par_1)):
        # print(i)
        y_center, x_center = par_1[img_i][0], par_1[img_i][1]
        for j in range(len(par_2)):
            b_center_x, b_center_y = par_2[j][0], par_2[j][1]
            # check if blob center is in circle
            if ((b_center_x - x_center) ** 2 + (b_center_y - y_center) ** 2) <= radius ** 2:
                #print('in circle')
                count.append(1)
    return sum(count)

# labelling frames before concatenation
def text_fmt (img_input , img_input_2, img_input_3, org=(350, 50), color=(125, 246, 55)):
    """this receives the imaages and output the frame title"""
    img = copy.deepcopy(img_input)
    img_1 = copy.deepcopy(img_input_2)
    img_2 = copy.deepcopy(img_input_3)
    new_img = cv2.putText(img=img, text= "Original Frame", org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=color, thickness=1 )
    new_img_1 = cv2.putText(img=img_1, text="CNN Output", org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=color, thickness=1)
    new_img_2 = cv2.putText(img=img_2, text="Final Output", org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=color, thickness=1)
    return (new_img, new_img_1, new_img_2)

def text_fmt_filler (img_input, text = "upper text", text_2 ='lower part',  org=( 450, 300), org_2 =(500, 350),  color=1, fontScale=1.0):
    """this receives the images and output the sub writing on the frames"""
    img = copy.deepcopy(img_input)
    new_img = cv2.putText(img=img, text= text, org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, color=color, thickness=1 )
    img = cv2.putText(img=new_img, text= text_2, org=org_2, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.3, color=color, thickness=2)
    return (img)

# image concatenation
def join_frames (original_frame, binary_img, img_pred):
    """This received all the images and join them together"""
    img = np.concatenate((original_frame, binary_img, img_pred), axis=1)
    return img

#%%
#complete code video crteation
#cleaning up directries from previous usgae
for f in os.listdir(file_ext_path):
    os.remove(file_ext_path + f)

print(len(os.listdir(file_ext_path)))
#splitting video files into frames
file_extract = VideoFrames(file = sample_video , rate = 50)
file_extract.extract_frames( out = file_ext_path)
avail_frame = len(os.listdir(file_ext_path))
print(avail_frame)
#%%
for file in range(len(os.listdir(file_ext_path))):
    print(file)
    presnt_name = file_ext_path + os.listdir(file_ext_path)[file]
    print(presnt_name)
    #file processing
    #fra = cv2.imread(presnt_name, 0)
    #plt.imshow(fra)
    #fra.shape
    encoder_img, orig_grey, orig_color = frame_processing(presnt_name)
    #CNN predictions and blob extractions0
    head_coor, body_coor, tail_coor = extract_blobs(encoder_img)
    #using prediction on the images
    #head part
    detection_img_hd, color_hd, cnn_img_hd, = draw_blob_wif_coor(given_blob  = head_coor,  img_color_in= orig_color )
    #body part
    detection_img_bd, color_bd, cnn_img_bd, = draw_blob_wif_coor(given_blob=body_coor, img_color_in=orig_color)
    # counting images in relevant functional areas..
    #feeder
    present_feeder_count = circle_count( head_coor, feeder, feeder_radius)
    present_drinker_count = circle_count(head_coor, drinker, drinker_radius)
    # image labels
    # top label
    presnt_orig, presnt_cnn_img, presnt_detect_img = text_fmt(orig_color, cnn_img_hd, detection_img_bd,  color=(1, 1, 1))
    # sub label of images
    presnt_detect_img_ =  text_fmt_filler(img_input=presnt_detect_img, text=f"Turkey Count", text_2=f"{len(head_coor)} ", fontScale=0.7, color=(1,1,0))
    presnt_cnn_img = text_fmt_filler(img_input=presnt_cnn_img, text=f"Drinker user", text_2=f"{present_drinker_count} ", fontScale=0.7, color=(1,1,0))
    presnt_cnn_img_ = text_fmt_filler (img_input=presnt_cnn_img, text=f"Feeder user", text_2 = f"{present_feeder_count} ", org=( 450, 150), org_2 =(500, 200), fontScale=0.7,  color=(1,1,0))
    #joining of images
    present_join = join_frames(presnt_orig, presnt_cnn_img_, presnt_detect_img_)
    joined_images.append(present_join)
    # images generated are saved in defined folder
    name = "/Home/makanji/Master_Thesis/Datasets/Created_test_color/" + "Joined_img_" + str(file) + ".jpg"
    saved_img = present_join * 255
    cv2.imwrite(name, saved_img)
    len(joined_images)
    # once its last images on the list, videos should be processed and saved
    if file == avail_frame-1:
        img_array = []
        vid_amount = 3
        for filename in os.listdir(path_extracted_pred):
            vid_img = cv2.imread(path_extracted_pred + filename)
            height, width, channel_no = vid_img.shape
            size = (width, height)
            img_array.append(vid_img)
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            vid_name = "/Home/makanji/Master_Thesis/Datasets/Video_out/color_model_performance.avi"
        out = cv2.VideoWriter(vid_name, fourcc, vid_amount, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

#%%
plt.imshow(joined_images[0])