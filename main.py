import cv2
import numpy as np
import math
from math import pi
from math import sqrt
import sys 
from keypoint_config import *
import XMLHandler
import front_image_keypoint_selector
import side_image_keypoint_selector
import os.path
my_path = os.path.abspath(os.path.dirname(__file__))

FRONT_KEYPOINTS = {"right_sholder_outer_point":[0,0], "right_sholder_inner_point":[0,0], "left_sholder_inner_point":[0,0], "left_sholder_outer_point":[0,0],
                "right_chest_point":[0,0], "left_chest_point":[0,0], "right_waist_point":[0,0], "left_waist_point":[0,0], "right_natural_waist_point":[0,0],
                "left_natural_waist_point":[0,0], "right_hip_point":[0,0], "left_hip_point":[0,0], "pants_crotch_point":[0,0], "right_thigh_point":[0,0],
                "left_thigh_point":[0,0], "right_Knee_point":[0,0], "left_Knee_point":[0,0], "right_ankle_point":[0,0], "left_ankle_point":[0,0],
                "head_point":[0,0], "foot_point":[0,0]} 

front_measurements_dict = {"front_sholder_width":[], "front_chest_width":[], "front_waist_width":[], "front_natural_waist_width":[], "front_hip_width":[],
                            "front_right_thigh_width":[], "front_left_thigh_width":[], "front_pants_crotch_right_knee_length":[], "front_pants_crotch_left_knee_width":[],
                            "front_right_knee_ankle_width":[], "front_left_knee_ankle_width":[], "front_shoulder_outer_waist":[], "front_shoulder_outer_waist_avg":[],
                            "front_shoulder_outer_natural_waist":[], "front_shoulder_outer_natural_waist_avg":[], "front_inseam_length":[], "front_inseam_length_avg":[]}

SIDE_KEYPOINTS = {"chest_front_point":[0,0], "chest_back_point":[0,0], "waist_front_point":[0,0], "waist_back_point":[0,0],"natural_waist_front_point":[0,0],
                    "natural_waist_back_point":[0,0], "mid_natural_waist_point":[0,0], "hip_front_point":[0,0], "hip_back_point":[0,0],"hip_mid_point":[0,0],"knee_point":[0,0],"ankle_point":[0,0], "shoulder_point":[0,0], "elbow_point":[0,0],
                    "wrest_point":[0,0], "head_point":[0,0], "foot_point":[0,0]}

side_measurements_dict = {"side_chest_width":[], "side_waist_width":[], "side_natural_waist_width":[], "side_hip_width":[], "side_hip_knee_length":[],"side_natural_waist_knee_length": [],
                          "side_knee_ankle_length":[], "side_sholder_elbow_legnth":[], "side_elbow_wrest_legnth":[], "side_full_arm_legnth":[], "side_full_leg_legnth":[], 
                          "side_shoulder_waist":[],"side_shoulder_natural_waist":[],"side_shoulder_waist_avg":[], "side_shoulder_natural_waist_avg":[]}  


girth_measurments_dict = {"chest":[], "waist":[], "natural_waist":[], "hip":[]} 


def rectangle_premiter(r1,r2):
    return 2*(r1+r2)

def circle_circumfrance(radius):
    return 2*pi*radius

def ellipse_perimeter1(r1,r2):

    permeter1 = (2 * pi * sqrt( (r1**2 + r2**2) / (2 * 1.0) ) ) 
    permeter2 = pi*(3*(r1+r2)-sqrt((3*r1+r2)*(r1+3*r2)))

    permeter = ((permeter1*1.02 + permeter2)/2)
    return permeter1

def ellipse_perimeter2(r1,r2):

    permeter1 = (2 * pi * sqrt( (r1**2 + r2**2) / (2 * 1.0) ) ) 
    permeter2 = pi*(3*(r1+r2)-sqrt((3*r1+r2)*(r1+3*r2)))

    permeter = ((permeter1*1.02 + permeter2)/2)
    return permeter2

def ellipse_perimeter(r1,r2):

    permeter1 = (2 * pi * sqrt( (r1**2 + r2**2) / (2 * 1.0) ) ) 
    permeter2 = pi*(3*(r1+r2)-sqrt((3*r1+r2)*(r1+3*r2)))

    permeter = ((permeter1*1.02 + permeter2)/2)
    return permeter


def apply_filter_return_countoures(hsv_image, hsv_color_lower, hsv_color_upper, min_contor_area = 15, img = None , contor_label=""):
    filtered_image = cv2.inRange(hsv_image, hsv_color_lower, hsv_color_upper)
    kernal = np.ones((2,2),"uint8")
    filtered_image = cv2.dilate(filtered_image,kernal)
    (contors,hierarchy) = cv2.findContours(filtered_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.bitwise_and(img, img, mask = filtered_image)
    if True:
        i = 0
        for pic, contour in enumerate(contors):
            area = cv2.contourArea(contour)
            if(area>min_contor_area):
                x,y,w,h = cv2.boundingRect(contour)
                cv2.putText(img,str(contor_label),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
                i += 1
            return (res, filtered_image, (x+3,y+3), hierarchy)

    return (res, filtered_image, contors, hierarchy)

def eclidian_distance(point1 , point2):
    global front_image_hip_hight, front_image_left_hip_pixel, front_image_right_hip_pixel
    (x1, y1) = point1
    (x2, y2) = point2
    result = 0
    mid = (front_image_left_hip_pixel[0]+front_image_right_hip_pixel[0]) / 2
    if y1 >= front_image_hip_hight and y2 >= front_image_hip_hight:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/front_image_lower_ratio
    elif y1 <= front_image_hip_hight and y2 <= front_image_hip_hight:
        return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/front_image_upper_ratio
    else:
        if y1 > front_image_hip_hight:
            if x1 > mid:
                result += ((math.sqrt(((x1-front_image_left_hip_pixel[0])**2)+((y1-front_image_left_hip_pixel[1])**2)))/front_image_lower_ratio)
            else:
                result += ((math.sqrt(((x1-front_image_right_hip_pixel[0])**2)+((y1-front_image_right_hip_pixel[1])**2)))/front_image_lower_ratio)
        else:
            if x1 > mid:
                result += ((math.sqrt(((x1-front_image_left_hip_pixel[0])**2)+((y1-front_image_left_hip_pixel[1])**2)))/front_image_upper_ratio)
            else:
                result += ((math.sqrt(((x1-front_image_right_hip_pixel[0])**2)+((y1-front_image_right_hip_pixel[1])**2)))/front_image_upper_ratio)
        if y2 > front_image_hip_hight:
            if x2 > mid:
                result += ((math.sqrt(((front_image_left_hip_pixel[0]-x2)**2)+((front_image_left_hip_pixel[1]-y2)**2)))/front_image_lower_ratio)
            else:
                result += ((math.sqrt(((front_image_right_hip_pixel[0]-x2)**2)+((front_image_right_hip_pixel[1]-y2)**2)))/front_image_lower_ratio)
        else:
            if x2 > mid:
                result += ((math.sqrt(((front_image_left_hip_pixel[0]-x2)**2)+((front_image_left_hip_pixel[1]-y2)**2)))/front_image_upper_ratio)
            else:
                result += ((math.sqrt(((front_image_right_hip_pixel[0]-x2)**2)+((front_image_right_hip_pixel[1]-y2)**2)))/front_image_upper_ratio)

    return result

def eclidian_distance2(point1 , point2, ratio):
    (x1, y1) = point1
    (x2, y2) = point2
    return (math.sqrt(((x1-x2)**2)+((y1-y2)**2)))/ratio

def eclidian_distance1D(point1 , point2, ratio):
    (x1, y1) = point1
    (x2, y2) = point2
    return (abs(y1-y2)/ratio)

def get_avg(in_list):

    return sum(in_list) / len(in_list) 

def front_image_measurments(front_image, points_dict):
    global FRONT_KEYPOINTS
    # sholder outer
    cv2.circle(front_image, points_dict["right_sholder_outer_point"], point_radius, right_sholder_outer_color, -1)
    #right sholder inner
    cv2.circle(front_image, points_dict["right_sholder_inner_point"], point_radius, right_sholder_inner_color, -1)
    #left sholder inner
    cv2.circle(front_image, points_dict["left_sholder_inner_point"], point_radius, left_sholder_inner_color, -1)
    #left sholder outer
    cv2.circle(front_image, points_dict["left_sholder_outer_point"], point_radius, left_sholder_outer_color, -1)

    #chest
    cv2.circle(front_image, points_dict["right_chest_point"], point_radius, right_chest_color, -1)
    cv2.circle(front_image, points_dict["left_chest_point"], point_radius, left_chest_color, -1)
    
    #waist
    cv2.circle(front_image, points_dict["right_waist_point"], point_radius, right_waist_color, -1)
    cv2.circle(front_image, points_dict["left_waist_point"], point_radius, left_waist_color, -1)

    #right chest and arm meeting
    cv2.circle(front_image, points_dict["right_natural_waist_point"], point_radius, right_natural_waist_color, -1)
    cv2.circle(front_image, points_dict["left_natural_waist_point"], point_radius, left_natural_waist_color, -1)
    
    #hip
    cv2.circle(front_image, points_dict["right_hip_point"], point_radius, right_hip_color, -1)
    cv2.circle(front_image, points_dict["left_hip_point"], point_radius, left_hip_color, -1)

    #pant crotch
    cv2.circle(front_image, points_dict["pants_crotch_point"], point_radius, pants_crotch_color, -1)
    
    #thigts
    cv2.circle(front_image, points_dict["right_thigh_point"], point_radius, right_thigh_color, -1)
    cv2.circle(front_image, points_dict["left_thigh_point"], point_radius, left_thigh_color, -1)
    
    #knees
    cv2.circle(front_image, points_dict["right_Knee_point"], point_radius, right_Knee_color, -1)
    cv2.circle(front_image, points_dict["left_Knee_point"], point_radius, left_Knee_color, -1)

    #ankles
    cv2.circle(front_image, points_dict["right_ankle_point"], point_radius, right_ankle_color, -1)
    cv2.circle(front_image, points_dict["left_ankle_point"], point_radius, left_ankle_color, -1)
    
    hsv = cv2.cvtColor(front_image, cv2.COLOR_BGR2HSV)
    #hsv hue set value  (color)
    for key in front_image_points_dict:
        FRONT_KEYPOINTS[key] = front_image_points_dict[key]
        

def side_image_measurments(side_image,points_dict):
    global SIDE_KEYPOINTS
    # chest
    cv2.circle(side_image, points_dict["chest_front_point"], point_radius, chest_front_color, -1)
    cv2.circle(side_image, points_dict["chest_back_point"], point_radius, chest_back_color, -1)
    
    # waist
    cv2.circle(side_image, points_dict["waist_front_point"], point_radius, waist_front_color, -1)
    cv2.circle(side_image, points_dict["waist_back_point"], point_radius, waist_back_color, -1)

    # natural_waist
    cv2.circle(side_image, points_dict["natural_waist_front_point"], point_radius, natural_waist_front_color, -1)
    cv2.circle(side_image, points_dict["natural_waist_back_point"], point_radius, natural_waist_back_color, -1)
    
    # hips
    cv2.circle(side_image, points_dict["hip_front_point"], point_radius, hip_front_color, -1)
    cv2.circle(side_image, points_dict["hip_back_point"], point_radius, hip_back_color, -1)
    
    #knee_point
    cv2.circle(side_image, points_dict["knee_point"], point_radius, knee_color, -1)
    
    #ankle
    cv2.circle(side_image, points_dict["ankle_point"], point_radius, ankle_color, -1)
    
    #shoulder
    cv2.circle(side_image, points_dict["shoulder_point"], point_radius, shoulder_color, -1)
    
    #elbow
    cv2.circle(side_image, points_dict["elbow_point"], point_radius, elbow_color, -1)
    
    #elbow
    cv2.circle(side_image, points_dict["wrest_point"], point_radius, wrest_color, -1)

    hsv = cv2.cvtColor(side_image, cv2.COLOR_BGR2HSV)
   
    for key in side_image_points_dict:
        SIDE_KEYPOINTS[key] = side_image_points_dict[key]
    SIDE_KEYPOINTS["hip_mid_point"][0] = (SIDE_KEYPOINTS["hip_front_point"][0] + SIDE_KEYPOINTS["hip_back_point"][0])/2
    SIDE_KEYPOINTS["hip_mid_point"][1] = (SIDE_KEYPOINTS["hip_front_point"][1] + SIDE_KEYPOINTS["hip_back_point"][1])/2
    SIDE_KEYPOINTS["mid_natural_waist_point"][0] = (SIDE_KEYPOINTS["natural_waist_front_point"][0] + SIDE_KEYPOINTS["natural_waist_back_point"][0])/2
    SIDE_KEYPOINTS["mid_natural_waist_point"][1] = (SIDE_KEYPOINTS["natural_waist_front_point"][1] + SIDE_KEYPOINTS["natural_waist_back_point"][1])/2
    

def resize_image(img,scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dsize = (width, height)
    # resize image
    img = cv2.resize(img, dsize)
    return img


def calulate_front_measurments(front_image_ratio):
    global FRONT_KEYPOINTS, front_measurements_dict 
    front_measurements_dict["front_sholder_width"].append(eclidian_distance2(FRONT_KEYPOINTS["left_sholder_outer_point"],FRONT_KEYPOINTS["right_sholder_outer_point"],front_image_ratio) )
    front_measurements_dict["front_chest_width"].append( eclidian_distance2(FRONT_KEYPOINTS["left_chest_point"],FRONT_KEYPOINTS["right_chest_point"],front_image_ratio) )
    front_measurements_dict["front_waist_width"].append( eclidian_distance2(FRONT_KEYPOINTS["left_waist_point"],FRONT_KEYPOINTS["right_waist_point"],front_image_ratio) )
    front_measurements_dict["front_natural_waist_width"].append( eclidian_distance2(FRONT_KEYPOINTS["left_natural_waist_point"],FRONT_KEYPOINTS["right_natural_waist_point"],front_image_ratio) )
    front_measurements_dict["front_hip_width"].append( eclidian_distance2(FRONT_KEYPOINTS["left_hip_point"],FRONT_KEYPOINTS["right_hip_point"],front_image_ratio) )
    front_measurements_dict["front_right_thigh_width"].append( eclidian_distance2(FRONT_KEYPOINTS["pants_crotch_point"],FRONT_KEYPOINTS["right_thigh_point"],front_image_ratio) )
    front_measurements_dict["front_left_thigh_width"].append( eclidian_distance2(FRONT_KEYPOINTS["pants_crotch_point"],FRONT_KEYPOINTS["left_thigh_point"],front_image_ratio) )
    front_measurements_dict["front_pants_crotch_right_knee_length"].append( eclidian_distance2(FRONT_KEYPOINTS["pants_crotch_point"],FRONT_KEYPOINTS["right_Knee_point"],front_image_ratio) )
    front_measurements_dict["front_pants_crotch_left_knee_width"].append( eclidian_distance2(FRONT_KEYPOINTS["pants_crotch_point"],FRONT_KEYPOINTS["left_Knee_point"],front_image_ratio) )
    front_measurements_dict["front_right_knee_ankle_width"].append( eclidian_distance2(FRONT_KEYPOINTS["right_Knee_point"],FRONT_KEYPOINTS["right_ankle_point"],front_image_ratio) )
    front_measurements_dict["front_left_knee_ankle_width"].append( eclidian_distance2(FRONT_KEYPOINTS["left_Knee_point"],FRONT_KEYPOINTS["left_ankle_point"],front_image_ratio) )
    
    front_measurements_dict["front_shoulder_outer_waist"].append(eclidian_distance1D(FRONT_KEYPOINTS["right_sholder_outer_point"],FRONT_KEYPOINTS["right_waist_point"],front_image_ratio))
    front_measurements_dict["front_shoulder_outer_waist"].append(eclidian_distance1D(FRONT_KEYPOINTS["left_sholder_outer_point"], FRONT_KEYPOINTS["left_waist_point"],front_image_ratio))
    front_measurements_dict["front_shoulder_outer_natural_waist"].append(eclidian_distance1D(FRONT_KEYPOINTS["right_sholder_outer_point"], FRONT_KEYPOINTS["right_natural_waist_point"],front_image_ratio))
    front_measurements_dict["front_shoulder_outer_natural_waist"].append(eclidian_distance1D(FRONT_KEYPOINTS["left_sholder_outer_point"], FRONT_KEYPOINTS["left_natural_waist_point"],front_image_ratio))

    front_measurements_dict["front_inseam_length"].append(front_measurements_dict["front_pants_crotch_right_knee_length"][0]+front_measurements_dict["front_right_knee_ankle_width"][0])
    front_measurements_dict["front_inseam_length"].append(front_measurements_dict["front_pants_crotch_left_knee_width"][0]+front_measurements_dict["front_left_knee_ankle_width"][0])
    
    front_measurements_dict["front_shoulder_outer_waist_avg"].append(get_avg(front_measurements_dict["front_shoulder_outer_waist"]))
    front_measurements_dict["front_shoulder_outer_natural_waist_avg"].append(get_avg(front_measurements_dict["front_shoulder_outer_natural_waist"]))
    front_measurements_dict["front_inseam_length_avg"].append(get_avg(front_measurements_dict["front_inseam_length"]))

def calulate_side_measurments(side_image_ratio):
    global SIDE_KEYPOINTS, side_measurements_dict
    
    side_measurements_dict["side_chest_width"].append(eclidian_distance2(SIDE_KEYPOINTS["chest_front_point"],SIDE_KEYPOINTS["chest_back_point"],side_image_ratio))
    side_measurements_dict["side_waist_width"].append(eclidian_distance2(SIDE_KEYPOINTS["waist_front_point"],SIDE_KEYPOINTS["waist_back_point"],side_image_ratio))
    side_measurements_dict["side_natural_waist_width"].append(eclidian_distance2(SIDE_KEYPOINTS["natural_waist_front_point"],SIDE_KEYPOINTS["natural_waist_back_point"],side_image_ratio))
    side_measurements_dict["side_hip_width"].append(eclidian_distance2(SIDE_KEYPOINTS["hip_front_point"],SIDE_KEYPOINTS["hip_back_point"],side_image_ratio))
    side_measurements_dict["side_hip_knee_length"].append(eclidian_distance2(SIDE_KEYPOINTS["hip_mid_point"],SIDE_KEYPOINTS["knee_point"],side_image_ratio))
    side_measurements_dict["side_natural_waist_knee_length"].append(eclidian_distance2(SIDE_KEYPOINTS["mid_natural_waist_point"],SIDE_KEYPOINTS["knee_point"],side_image_ratio))
    side_measurements_dict["side_knee_ankle_length"].append(eclidian_distance2(SIDE_KEYPOINTS["knee_point"],SIDE_KEYPOINTS["ankle_point"],side_image_ratio))
    side_measurements_dict["side_sholder_elbow_legnth"].append(eclidian_distance2(SIDE_KEYPOINTS["shoulder_point"],SIDE_KEYPOINTS["elbow_point"],side_image_ratio))
    side_measurements_dict["side_elbow_wrest_legnth"].append(eclidian_distance2(SIDE_KEYPOINTS["elbow_point"],SIDE_KEYPOINTS["wrest_point"],side_image_ratio))
    side_measurements_dict["side_full_arm_legnth"].append(side_measurements_dict["side_sholder_elbow_legnth"][0] + side_measurements_dict["side_elbow_wrest_legnth"][0])
    side_measurements_dict["side_full_leg_legnth"].append(side_measurements_dict["side_hip_knee_length"][0] + side_measurements_dict["side_knee_ankle_length"][0])
    side_measurements_dict["side_full_leg_legnth"].append(side_measurements_dict["side_natural_waist_knee_length"][0] + side_measurements_dict["side_knee_ankle_length"][0])
    
    side_measurements_dict["side_shoulder_waist"].append(eclidian_distance1D(SIDE_KEYPOINTS["shoulder_point"], SIDE_KEYPOINTS["waist_front_point"],side_image_ratio))
    side_measurements_dict["side_shoulder_waist"].append(eclidian_distance1D(SIDE_KEYPOINTS["shoulder_point"], SIDE_KEYPOINTS["waist_back_point"],side_image_ratio))

    side_measurements_dict["side_shoulder_natural_waist"].append(eclidian_distance1D(SIDE_KEYPOINTS["shoulder_point"], SIDE_KEYPOINTS["natural_waist_front_point"],side_image_ratio))
    side_measurements_dict["side_shoulder_natural_waist"].append(eclidian_distance1D(SIDE_KEYPOINTS["shoulder_point"], SIDE_KEYPOINTS["natural_waist_back_point"],side_image_ratio))
    
    side_measurements_dict["side_shoulder_waist_avg"].append(get_avg(side_measurements_dict["side_shoulder_waist"]))
    side_measurements_dict["side_shoulder_natural_waist_avg"].append(get_avg(side_measurements_dict["side_shoulder_natural_waist"]))

def girth_calulate_helper(r1, r2,measurment_key):
    global girth_measurments_dict
    girth_measurments_dict[measurment_key].append(ellipse_perimeter2(r1,r2))
    girth_measurments_dict[measurment_key].append((ellipse_perimeter1(r1,r2)/2)+ellipse_perimeter1(r1,r2/2)/2)
    
def calulate_girth_measurements():
    global FRONT_KEYPOINTS, SIDE_KEYPOINTS, front_measurements_dict, side_measurements_dict, true_measurments_dict
    
    girth_calulate_helper(front_measurements_dict["front_chest_width"][0]/2,side_measurements_dict["side_chest_width"][0]/2,"chest")
    girth_calulate_helper(front_measurements_dict["front_waist_width"][0]/2,side_measurements_dict["side_waist_width"][0]/2,"waist")
    girth_calulate_helper(front_measurements_dict["front_natural_waist_width"][0]/2,side_measurements_dict["side_natural_waist_width"][0]/2,"natural_waist")
    girth_calulate_helper(front_measurements_dict["front_hip_width"][0]/2,side_measurements_dict["side_hip_width"][0]/2,"hip")


if __name__ == "__main__":


    print(len(sys.argv))
    if len(sys.argv) != 7:
        print("script must be called like: python <scriptName> <sperson> <user's height> <users' gender> <front image path> <side image path> <option>")
        sys.exit()
    else:
        if not sys.argv[2].isnumeric():
            print("script must be called like: python <scriptName> <sperson> <user's height> <users' gender> <front image path> <side image path> <option>")
            sys.exit()
    
    user_name = sys.argv[1]
    user_height = float(sys.argv[2])
    user_gender = sys.argv[3]
    front_image_name = sys.argv[4]
    side_image_name = sys.argv[5]
    option = int(sys.argv[6])
    
    # user_name = "Rimon"
    # user_height = 168
    # user_gender = "male"
    # front_image_name = "images/Rimon1.1.jpg"
    # side_image_name = "images/Rimon1.2.jpg"
    # option = 0
    
    if option == 0:
        front_image_keypoint_selector.main(user_name,user_height,front_image_name)
        side_image_keypoint_selector.main(user_name,user_height,side_image_name)

    front_image_points_dict = XMLHandler.read_file(os.path.join(my_path,"front_keypoints/"+user_name+"_front"))
    front_image_points_dict = XMLHandler.refract_dict(front_image_points_dict)
    
    side_image_points_dict = XMLHandler.read_file(os.path.join(my_path,"side_keypoints/"+user_name+"_side"))
    side_image_points_dict = XMLHandler.refract_dict(side_image_points_dict)
   

    print("front_image_points_dict = ",front_image_points_dict)
    print("side_image_points_dict = ",side_image_points_dict)
    
    front_image = cv2.imread(front_image_name, cv2.IMREAD_COLOR)
    # scaling 
    print("front_image.shape[1]",front_image.shape[1])
    print("front_image.shape[0]",front_image.shape[0])
    #calculate the 50 percent of original dimensions
    front_image = resize_image(front_image, front_image_points_dict["front_image_scale_percent"][0])
    print("front_image.shape[1]",front_image.shape[1])
    print("front_image.shape[0]",front_image.shape[0])
    # end of scalling
    
    side_image = cv2.imread(side_image_name, cv2.IMREAD_COLOR)
    # scaling 
    print("side_image.shape[1]",side_image.shape[1])
    print("side_image.shape[0]",side_image.shape[0])
    #calculate the 50 percent of original dimensions
    side_image = resize_image(side_image, side_image_points_dict["side_image_scale_percent"][0])
    print("side_image.shape[1]",side_image.shape[1])
    print("side_image.shape[0]",side_image.shape[0])
    # end of scalling


    front_image_measurments(front_image, front_image_points_dict)
    side_image_measurments(side_image, side_image_points_dict)

    front_image_ratio = front_image_points_dict["front_image_upper_ratio"][0]
    side_image_ratio = side_image_points_dict["side_image_upper_ratio"][0]
  
    calulate_front_measurments(front_image_ratio)
    calulate_side_measurments(side_image_ratio)
    calulate_girth_measurements()
    
    for key in front_measurements_dict:
        print("key -> ", key ,"  ",front_measurements_dict[key])
    print("side image measurments ")
    for key in side_measurements_dict:
        print("key -> ", key ,"  ",side_measurements_dict[key])

    for key in girth_measurments_dict:
        print("key -> ", key ,"  ",girth_measurments_dict[key])
    
    result_dict = {}
    result_dict["name"] = user_name
    result_dict["gender"] = user_gender
    result_dict["height"] = user_height

    # result_dict.update(front_measurements_dict)
    # result_dict.update(side_measurements_dict)
    # result_dict.update(girth_measurments_dict)

    for key in front_measurements_dict:
        result_dict[key] = front_measurements_dict[key][0]
    for key in side_measurements_dict:
        result_dict[key] = side_measurements_dict[key][0]
    
    result_dict["side_full_leg_legnth"] = side_measurements_dict["side_full_leg_legnth"][1]+4
    for key in girth_measurments_dict:
        result_dict[key] = girth_measurments_dict[key][0]
    
    print(result_dict)
    
    XMLHandler.write(result_dict,os.path.join(my_path,"results/"+user_name))
    cv2.imshow("front_image", front_image)   
    cv2.imshow("side_image", side_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
