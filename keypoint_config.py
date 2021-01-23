''' Configuration
    Date: 26 NOV 2019
    Author: Rimon Adel
'''
import numpy as np

point_radius = 3

right_sholder_outer_color = (0,0,255)
right_sholder_inner_color = (18,18,224)
left_sholder_inner_color = (26,26,199)
left_sholder_outer_color = (3,255,11)

right_chest_color = (0,0,173)
left_chest_color = (28,28,163)

right_waist_color = (42,42,163)
left_waist_color = (117,255,237)

right_natural_waist_color = (221,0,255)
left_natural_waist_color = (197,20,224)

right_hip_color = (239,133,255)
left_hip_color = (242,161,225)

pants_crotch_color = (227,48,255)

right_thigh_color = (234,97,255)
left_thigh_color = (255,0,149)

right_Knee_color = (255,25,159)
left_Knee_color = (255,66,255)

right_ankle_color = (255,115,197)
left_ankle_color = (255,145,231)

Head_color = (255,0,255)
foot_color = (169,245,235)

#hsv hue set value  (color)
left_sholder_inner_lower = np.array([-10, 212, 159])
left_sholder_inner_upper = np.array([ 10, 232, 239])

left_sholder_outer_lower = np.array([ 49, 242, 215])
left_sholder_outer_upper = np.array([ 69, 262, 295])

right_sholder_inner_lower = np.array([-10, 225, 184])
right_sholder_inner_upper = np.array([ 10, 245, 264])

right_sholder_outer_lower = np.array([-10, 245, 215])
right_sholder_outer_upper = np.array([ 10, 265, 295])

right_chest_lower = np.array([-10, 179, 123])
right_chest_lupper = np.array([ 10, 199, 203])

left_chest_lower = np.array([ 24, 128, 215])
left_chest_upper = np.array([ 44, 148, 295])

right_waist_lower = np.array([ -10, 245, 133])
right_waist_upper = np.array([ 10, 265, 213])

left_waist_lower = np.array([ -10, 201, 123])
left_waist_upper = np.array([ 10, 221, 203])

right_natural_waist_lower = np.array([ 144, 222, 184])
right_natural_waist_upper = np.array([ 164, 242, 264])

left_natural_waist_lower = np.array([ 144, 245, 215])
left_natural_waist_upper = np.array([ 164, 265, 295])

right_hip_lower = np.array([ 144,112,215])
right_hip_upper = np.array([ 164,132,295])

left_hip_lower = np.array([ 134, 75, 202])
left_hip_upper = np.array([ 154, 95, 282])

pants_crotch_lower = np.array([ 144, 148, 215])
pants_crotch_upper = np.array([ 164, 168, 295])

right_thigh_lower = np.array([ 144, 197, 215])
right_thigh_upper = np.array([ 164, 217, 295])

left_thigh_lower = np.array([ 127, 220, 215])
left_thigh_upper = np.array([ 147, 240, 295])

right_Knee_lower = np.array([ 128, 245, 215])
right_Knee_upper = np.array([ 148, 265, 295])

left_Knee_lower = np.array([ 140, 179, 215])
left_Knee_upper = np.array([ 160, 199, 295])

right_ankle_lower = np.array([ 44,  33, 212])
right_ankle_upper = np.array([ 64,  53, 292]) 

left_ankle_lower = np.array([ 56,  66, 212]) 
left_ankle_upper = np.array([ 76,  86, 292])


#####################################################################################################################################
### side view  image 

chest_front_color = (239,133,255)
chest_back_color = (242,161,225)

waist_front_color = (227,48,255)
waist_back_color = (234,97,255)

natural_waist_front_color = (255,0,149)
natural_waist_back_color = (255,25,159)

hip_front_color = (255,115,197)
hip_back_color = (255,145,231)

knee_color = (255,115,197)

ankle_color = (169,245,235)

shoulder_color = (0,0,255)

elbow_color = (255,66,255)

wrest_color = (217,247,139)



# filters 
chest_front_lower = np.array([ 144,112,215])
chest_front_upper = np.array([ 164,132,295])

chest_back_lower =  np.array([ 134, 75, 202])
chest_back_upper = np.array([ 154, 95, 282])

waist_front_lower = np.array([ 144, 148, 215])
waist_front_upper = np.array([ 164, 168, 295])

waist_back_lower = np.array([ 144, 197, 215])
waist_back_upper = np.array([ 164, 217, 295])

hip_front_lower = np.array([ 127, 220, 215])
hip_front_upper = np.array([ 147, 240, 295])

hip_back_lower = np.array([ 128, 245, 215])
hip_back_upper = np.array([ 148, 265, 295])

butt_front_lower =  np.array([ 133, 100, 215])
butt_front_upper =  np.array([ 153, 120, 295])

butt_back_lower =  np.array([ 128, 130, 215])
butt_back_upper =  np.array([ 148, 150, 295])
#####################################################################################################################################