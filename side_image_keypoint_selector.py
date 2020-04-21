import math
import cv2
import numpy as np
import math
from keypoint_config import *
import XMLHandler
import sys


points_dict = {}
result_dict={}

OUTPUT_DIRECTORY = "side_keypoints/"
## model img
MODEL_IMAGE_NAME = "Assets/model2.jpg"
## model image data
MODEL_IAMGE_DATA_FILE_NAME = "Assets/model2"
MODEL_POINTS_DICT = XMLHandler.read_file(MODEL_IAMGE_DATA_FILE_NAME)
MODEL_POINTS_DICT = XMLHandler.refract_dict(MODEL_POINTS_DICT)

iterator = 0

POINTS_COLOR = {0 : chest_front_color, 1 : chest_back_color, 2 : waist_front_color, 3 : waist_back_color, 4 : hip_front_color,
                5 : hip_back_color, 6 : trousers_front_color, 7 : trousers_back_color, 8 : Head_color, 9 : foot_color}

POINTS_NAMES = {0 : "chest_front_point", 1 : "chest_back_point", 2 : "waist_front_point", 3 : "waist_back_point", 4 : "hip_front_point", 
                5 : "hip_back_point", 6 : "trousers_front_point", 7 : "trousers_back_point", 8 : "head_point", 9 : "foot_point"}


image_width = 0 #image width to reject out of bound clicks

def eclidian_distance(point1 , point2):
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))
        
def draw_circle(event,x,y,flags,param):
    global points_dict, iterator, image_width
    if event == cv2.EVENT_LBUTTONDOWN and x <= image_width:
        points_dict[iterator] = (x,y)
        iterator += 1
        if iterator in POINTS_NAMES:
            print("now select ---> ",POINTS_NAMES[iterator],(x,y))
        else:
            print("one left click more to save")
    if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN :
        iterator -= 1
        print("now select ---> ",POINTS_NAMES[iterator])
        del points_dict[iterator]
     
def main(user_name, person_height, image_path):
    global POINTS_DICT, iterator, image_width
    file_name = user_name + "_side"
    print(file_name)
    
    file_name = OUTPUT_DIRECTORY+file_name
    
    mode_img = cv2.imread(MODEL_IMAGE_NAME, cv2.IMREAD_COLOR)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print("now select ---> ",POINTS_NAMES[iterator])

    
    SCALE_PERCENT = float((mode_img.shape[0] / img.shape[0])*100) # calculate needed Scale percent
    # scaling 
    width = int(img.shape[1] * SCALE_PERCENT / 100)
    height = int(img.shape[0] * SCALE_PERCENT / 100)
    dsize = (width, height)
    image_width = width 
    # resize image
    img = cv2.resize(img, dsize)
    
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
                model_image_marker_radius ^= 8
            timer += 1
            cv2.circle(mode_img, MODEL_POINTS_DICT[POINTS_NAMES[iterator]], model_image_marker_radius, POINTS_COLOR[iterator], -1)
            for point in points_dict:
                cv2.circle(img, points_dict[point], point_radius, POINTS_COLOR[point], -1)
                cv2.circle(mode_img, MODEL_POINTS_DICT[POINTS_NAMES[point]], point_radius+5, POINTS_COLOR[point], -1)
            img_concate_Verti=np.concatenate((img,mode_img),axis=1)
            cv2.imshow("test",img_concate_Verti)
            cv2.setMouseCallback('test',draw_circle)
            cv2.waitKey(1)
    except:
        cv2.destroyAllWindows()
        pass
    finally:
        ratio = eclidian_distance(points_dict[8],points_dict[9])/person_height
        result_dict["side_image_name"] = str(image_path)
        result_dict["side_image_scale_percent"] = str(SCALE_PERCENT)
        result_dict["side_image_upper_ratio"] = str(ratio)
        result_dict["side_image_upper_ratio"] = str(ratio)
        result_dict["side_image_ratio"] = str(ratio)
        for point in points_dict:
            result_dict[POINTS_NAMES[point]] = str(str(points_dict[point]))

        result_dict["side_image_HIP_HIGHT"] = str(points_dict[4][1])
        cv2.destroyAllWindows()
        XMLHandler.write(result_dict,file_name)
        return result_dict
        

if __name__ == "__main__":
    
    main("images/kishk1.2.jpg","kishk",179)
