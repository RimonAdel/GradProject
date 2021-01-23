import math
import cv2
import numpy as np
import math
from keypoint_config import *
import XMLHandler
import time
import sys
import os.path
my_path = os.path.abspath(os.path.dirname(__file__))

POINTS_DICT = {}
result_dict={}

OUTPUT_DIRECTORY = os.path.join(my_path, "front_keypoints/")

## model img
MODEL_IMAGE_NAME = os.path.join(my_path, "Assets/model1.jpg") 
## model image data
MODEL_IAMGE_DATA_FILE_NAME = os.path.join(my_path, "Assets/model1") 
MODEL_POINTS_DICT = XMLHandler.read_file(MODEL_IAMGE_DATA_FILE_NAME)
MODEL_POINTS_DICT = XMLHandler.refract_dict(MODEL_POINTS_DICT)

iterator = 0

POINTS_COLOR = {0 : right_sholder_outer_color, 1 : right_sholder_inner_color, 2 : left_sholder_inner_color, 3 : left_sholder_outer_color, 4 : right_chest_color,
                5 : left_chest_color, 6 : right_waist_color, 7 : left_waist_color, 8 : right_natural_waist_color, 9 : left_natural_waist_color, 10 : right_hip_color,
                11 : left_hip_color, 12 : pants_crotch_color, 13 : right_thigh_color, 14 : left_thigh_color, 15 : right_Knee_color, 16 : left_Knee_color,
                17 : right_ankle_color, 18 : left_ankle_color, 19 : Head_color, 20 : foot_color}

POINTS_NAMES = {0 : "right_sholder_outer_point", 1 : "right_sholder_inner_point", 2 : "left_sholder_inner_point", 3 : "left_sholder_outer_point", 4 : "right_chest_point", 
                5 : "left_chest_point", 6 : "right_waist_point", 7 : "left_waist_point", 8 : "right_natural_waist_point", 9 : "left_natural_waist_point",
                10 : "right_hip_point", 11 : "left_hip_point", 12 : "pants_crotch_point", 13 : "right_thigh_point", 14 : "left_thigh_point", 15 : "right_Knee_point",
                16 : "left_Knee_point", 17 : "right_ankle_point", 18 : "left_ankle_point", 19 : "head_point", 20 : "foot_point"}

image_width = 0 #image width to reject out of bound clicks

def eclidian_distance(point1 , point2):
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))

def draw_circle(event,x,y,flags,param):
    global POINTS_DICT, iterator, image_width
    
    if event == cv2.EVENT_LBUTTONDOWN and x <= image_width:
        POINTS_DICT[iterator] = (x,y)
        iterator += 1
        if iterator in POINTS_NAMES:
            print("now select ---> ",POINTS_NAMES[iterator],(x,y))
        else:
            print("one left click more to save")

    if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN :

        iterator -= 1
        print("now select ---> ",POINTS_NAMES[iterator])
        del POINTS_DICT[iterator]



def main(user_name, person_height, image_path):
    global POINTS_DICT, iterator, image_width

    file_name = user_name + "_front"
        
    file_name = OUTPUT_DIRECTORY+file_name
    
    mode_img = cv2.imread(MODEL_IMAGE_NAME, cv2.IMREAD_COLOR)
 
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(img)
    print("now select ---> ",POINTS_NAMES[iterator])
    
    SCALE_PERCENT = float(((mode_img.shape[0]) / img.shape[0])*100) # calculate needed Scale percent
    # scaling 
    width = int(img.shape[1] * (SCALE_PERCENT / 100))
    height = int(img.shape[0] * (SCALE_PERCENT / 100))
    dsize = (width, height)
    image_width = width 
    # resize image
    img = cv2.resize(img, dsize)
    # mode_img = cv2.resize(mode_img,( mode_img.shape[1], height))
    
    if mode_img.shape[0]>img.shape[0]:
        zeros_img = np.zeros((mode_img.shape[0],img.shape[1],3),dtype=np.uint8)
        zeros_img[:img.shape[0],:img.shape[1],:] = img
        img = zeros_img
    elif mode_img.shape[0] < img.shape[0]:     # i think this case will never happen because SCALE_PERCENT always results the floor of the equation 
        zeros_img = np.zeros((img.shape[0],mode_img.shape[1],3),dtype=np.uint8)
        zeros_img[:mode_img.shape[0],:mode_img.shape[1],:] = mode_img
        mode_img = zeros_img
        
    # pepare variables for the loop
    true_mode_img = mode_img.copy()
    true_image = img.copy()
    timer = 0
    model_image_marker_radius = 7

    try:
        while True:
            img = true_image.copy()
            mode_img = true_mode_img.copy()
            if timer >= 100:
                timer = 0
                model_image_marker_radius ^= 8  # 8 = 0b1000  7 = 0b0111 then 7^8 = 1111 or 1000

            timer += 1
            tup = (int(MODEL_POINTS_DICT[POINTS_NAMES[iterator]][0] ),int(MODEL_POINTS_DICT[POINTS_NAMES[iterator]][1]))
            cv2.circle(mode_img, tup, model_image_marker_radius, POINTS_COLOR[iterator], -1)
            for point in POINTS_DICT:
                cv2.circle(img, POINTS_DICT[point], point_radius, POINTS_COLOR[point], -1)
                tup = (int(MODEL_POINTS_DICT[POINTS_NAMES[point]][0]),int(MODEL_POINTS_DICT[POINTS_NAMES[point]][1]))
                cv2.circle(mode_img, tup, point_radius+5, POINTS_COLOR[point], -1)
            cv2.imshow("img",img)
            cv2.imshow("img_model",mode_img)
            cv2.setMouseCallback('img',draw_circle,image_width)
            cv2.waitKey(1)

    except:
        print("yess", iterator)
        pass
    finally:
        ratio = eclidian_distance(POINTS_DICT[len(POINTS_NAMES)-2],POINTS_DICT[len(POINTS_NAMES)-1])/person_height
        result_dict["front_image_name"] = str(image_path)
        result_dict["front_image_scale_percent"] = str(SCALE_PERCENT)
        result_dict["front_image_upper_ratio"] = str(ratio)
        result_dict["front_image_lower_ratio"] = str(ratio)
        result_dict["front_image_ratio"] = str(ratio)
        for point in POINTS_DICT:
            result_dict[POINTS_NAMES[point]] = str(POINTS_DICT[point])
        result_dict["front_image_hip_hight"] = str(POINTS_DICT[8][1])
        result_dict["front_image_right_hip_pixel"] = str(POINTS_DICT[8])
        result_dict["front_image_left_hip_pixel"] = str(POINTS_DICT[9])
        cv2.destroyAllWindows()
        XMLHandler.write(result_dict,file_name)
        return result_dict

        
if __name__ == "__main__":

    main("deadpool",180,"images/100.jpg")
