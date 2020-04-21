
def write(input_dict,filename):
    with open(filename+".xml", 'w') as the_file:
        the_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        the_file.write("<person>\n")
        for key in input_dict:
            the_file.write("\t<{key}>{value}</{key}>\n".format(key=key,value=input_dict[key]))
        the_file.write("</person>\n")
        
def read_file(filename):
    results_dict = {}
    with open(filename + ".xml", 'r') as the_file:
        iterartor = -1
        for line in the_file:
            iterartor += 1
            if iterartor == 0:
                continue
            line = line.replace("\t",'')
            ' '.join(line.split())
            if line.find('<',2) == -1:
                continue
            line = line[0:line.find('<',2)].replace('<','')
            line = line.split('>')
            results_dict[line[0]] = line[1]
    return results_dict

def refract_dict(result_dict):
    iterator = -1
    for key in result_dict:
        iterator += 1
        value = result_dict[key].replace(')','').replace('(','').split(',')
        if len(value) > 1:
            result_dict[key] = tuple([int(value[0]),int(value[1])])
        elif len(value) == 1 and iterator > 0:
            result_dict[key] = tuple([float(value[0])])
            
    return result_dict