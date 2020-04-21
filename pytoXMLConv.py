from keypoint_config import *
from side_keypoints.Rimon2_2 import *
import XMLHandler
to_write_Dict = {}
with open("side_keypoints/kisk2_2" + ".py", 'r') as the_file:
    iterartor = -1
    for line in the_file:
        line = line.replace('\n','')
        line = line.split('=')
        print(line, len(line))
        if len(line) > 1:
            to_write_Dict[line[0].replace(' ','')] = line[1].replace(' ','')

    for key in to_write_Dict:
        print(key, to_write_Dict[key])
    
    XMLHandler.write(to_write_Dict,"side_keypoints/kisk2_2")