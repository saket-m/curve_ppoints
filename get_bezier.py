from poha import *
from pista import *
from gen_spline import *
import os
import cv2
import sys

if __name__ == '__main__':
    '''
    sys.argv[1] = RGB mask path.
    sys.argv[2] = dir path where you want to store beziers.
    sys.argv[3] = algorithm used to derive vertices
                    0. random
                    1. if 1 then, cosidering slope, if 0 then, doesn't consider slope
    sys.argv[4] = if 3 is '1' then 2 specifies the slope thresold

    e.g. python get_bezier.py roto-14.png masks/test1/ 1 25
    '''
    rgb_mask_path = sys.argv[1]
    dir_name = sys.argv[2]

    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    
    rgb_mask = cv2.imread(rgb_mask_path)
    labels = get_labels(rgb_mask)

    i = 1
    for label in labels:
        label_mask = isolate_class(rgb_mask, label)
        cv2.imwrite(dir_name+'mask'+str(i)+'.png', label_mask)
        i += 1
    # resize_all(dir_name, (1080, 1920), f=False, grey=True)

    orig_list = []
    k = 500
    for image_name in os.listdir(dir_name):
        if image_name.split('.')[-1] == 'png':
            k = get_individual_mask(dir_name + image_name, dir_name, k=k)
            # os.remove(dir_name + image_name)
            orig_list.append(dir_name + image_name)

    remove_same_from_list(dir_name, orig_list)
    remove_from_list(dir_name, orig_list)
    remove_empty(dir_name)
    remove_same(dir_name)
    remove_low_area_mask(dir_name)
    # remove_superset(dir_name)

    for mask_path in os.listdir(dir_name):
        if not os.path.isfile(dir_name + mask_path):
            continue

        mask_num = int(mask_path.split('.')[0][4:])
        mask_path = dir_name + mask_path

        if int(sys.argv[3]) == 0:
            vertices = get_vertices(mask_path, 5)
        elif int(sys.argv[3]) == 1:
            if len(sys.argv) > 3:
                vertices = get_vertices_by_m(mask_path, t=float(sys.argv[4]))
            else:
                vertices = get_vertices_by_m(mask_path, t=25)

        unzip_vertices = ([i for i, j in vertices], [j for i, j in vertices])
        mask = cv2.imread(mask_path, 0)        ## load the image mask

        print(vertices)
        # print(v)

        with open(dir_name + 'strokes'+str(mask_num)+'.bezier', 'w') as f:
            for vertex in vertices:
                f.write(str(vertex[0])+'\t'+str(vertex[1])+'\n')
