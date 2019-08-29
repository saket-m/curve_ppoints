'''
This programme draws spline out of segmentation mask.

Arguments ::
    1. image_path
'''
import sys
import math

from pprint import pprint

import numpy as np
import cv2
from matplotlib import pyplot as plt

def get_outline(mask_path):
    '''
    Extracts the indices of the bordering pixels from a mask.

    Arguments ::
        mask_path -- str | path where the mask is stored
    Return ::
        outline -- tupple | outline pixel indices of the mask
    '''
    mask = cv2.imread(mask_path, 0)       ## load the image mask
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    if np.max(mask) == 255:
        mask = mask / 255
    
    mask = np.rot90(np.rot90(np.rot90(mask)))

    area = np.where(mask == 1) ## indices of the pixels involved in mask
    outline = []                        ## indices of the outline pixels of the mask
    idxs = list(zip(area[0], area[1]))  ## indices of the pixels involves in mask in coordinate form
    idx_dict = {}                       ## indices of the outline pixels of the mask in dictionary form
    unzip_outline = []

    for idx in idxs:
        if (idx[0] + 1, idx[1]) not in idxs or (idx[0] - 1, idx[1]) not in idxs or (idx[0] , idx[1] + 1) not in idxs or (idx[0], idx[1] - 1) not in idxs:
            if (idx[0] + 1, idx[1]) not in idxs:
                outline.append((idx[0] + 1, idx[1]))
            elif (idx[0] - 1, idx[1]) not in idxs:
                outline.append((idx[0] - 1, idx[1]))
            elif (idx[0] , idx[1] + 1) not in idxs:
                outline.append((idx[0] , idx[1] + 1))
            elif (idx[0] , idx[1] - 1) not in idxs:
                outline.append((idx[0] , idx[1] - 1))
    unzip_outline = ([i for i, j in outline], [j for i, j in outline])

    mask[area] = 0
    mask[unzip_outline] = 1
    cv2.imwrite('outline.png', mask)

    return tuple(outline)

def sort_outline(mask_path):
    '''
    Sorts the outkline in order.

    Arguments ::
        mask_path -- str | path where the mask is stored
    '''
    outline = get_outline(mask_path)
    if len(outline) > 1:
        sorted_outline = [outline[0], outline[1]]
        for i in range(1, len(outline)):
            dist = []
            p = []
            for j in range(1, len(outline)):
               if outline[j] not in sorted_outline:
                    dist.append(np.sqrt((outline[j][0] - sorted_outline[-1][0]) ** 2 + ((outline[j][1] - sorted_outline[-1][1]) ** 2)))
                    p.append(outline[j])
            if len(dist) > 0:
                idx = dist.index(min(dist))
                sorted_outline.append(p[idx])

        return sorted_outline

def get_m(mask_path, stride=4):
    '''
    Calculates slope at each point of the mask.

    Arguments ::
        mask_path -- str | path where the mask is stored
        stride -- int | number of pixels in between 2 vertices
    Return ::
        m_list -- list | contains a point and its corresponding slope
    '''
    m_list = []
    sorted_outline = sort_outline(mask_path)

    # m_dict[sorted_outline[0]] = (sorted_outline[0][1] - sorted_outline[1][1]) / (sorted_outline[0][0] - sorted_outline[1][0])
    for i in range(0, len(sorted_outline)):
        if i < len(sorted_outline)-stride:
            if (sorted_outline[i][0] - sorted_outline[i+stride][0]) == 0:
                m_list.append((sorted_outline[i], math.degrees(math.atan(abs(((sorted_outline[i][1] - sorted_outline[i+stride][1]) / 0.001))))))
            else:
                m_list.append((sorted_outline[i], math.degrees(math.atan(abs(((sorted_outline[i][1] - sorted_outline[i+stride][1]) / (sorted_outline[i][0] - sorted_outline[i+stride][0])))))))
        else:
            if (sorted_outline[i][0] - sorted_outline[-1 * (len(sorted_outline)-i-stride)][0]) == 0:
                m_list.append((sorted_outline[i], math.degrees(math.atan(abs(((sorted_outline[i][1] - sorted_outline[-1 * (len(sorted_outline)-i-stride)][1]) / 0.001))))))
            else:
                m_list.append((sorted_outline[i], math.degrees(math.atan(abs(((sorted_outline[i][1] - sorted_outline[-1 * (len(sorted_outline)-i-stride)][1]) / (sorted_outline[i][0] - sorted_outline[-1 * (len(sorted_outline)-i-stride)][0])))))))
    
    return m_list

def get_vertices_by_m(mask_path, t=25):
    '''
    Computes the cordinates of the poins to be used in the spline.

    Arguments ::
        mask_path -- str | path where the mask is stored
    Return ::
        vertices -- list | outline pixel indices of the mask
    '''
    print('processing', mask_path)

    m_list = get_m(mask_path, stride=4)
    print(m_list)
    vertices = []

    vertices.append(m_list[0][0])
    m1 = m_list[0][1]
    for vertex, m2 in m_list:
        if abs(m1 - m2) > t:
            vertices.append(vertex)
            m1 = m2

    return vertices


def get_vertices(mask_path, stride=5):
    '''
    Computes the cordinates of the poins to be used in the spline.

    Arguments ::
        mask_path -- str | path where the mask is stored
        stride -- int | number of pixels in between 2 vertices
    Return ::
        vertices -- list | outline pixel indices of the mask
    '''
    outline = get_outline(mask_path)
    vertices = []
    clockwise_vertices = []

    for i in range(len(outline)):
        if i % int(stride) == 0:
            vertices.append(outline[i])

    lower = []
    upper = [vertices[0]]
    for i in range(1, len(vertices)):
        if vertices[i][1] < vertices[i-1][1]:
            lower.append(vertices[i])
        else:
            upper.append(vertices[i])

    inline = [vertices[0], vertices[1]]
    for i in range(1, len(vertices)):
        dist = []
        p = []
        for j in range(1, len(vertices)):
            if vertices[j] not in inline:
                dist.append(np.sqrt((vertices[j][0] - inline[-1][0]) ** 2 + ((vertices[j][1] - inline[-1][1]) ** 2))) 
                p.append(vertices[j])
        if len(dist) > 0:
            idx = dist.index(min(dist))
            inline.append(p[idx])

    # return upper + lower[::-1]
    return inline

def sort_clockwise(vertices):
    '''
    Given a list of vertices sorts them in clockwise manner.

    Argumets ::
        vertices -- list | list of vertices (x, y) pair
    Return ::
        clockwise_vertices -- list | sorted list of vertices (x, y) pair
    '''
    clockwise_vertices = []
    
    x_pre, y_pre = vertices[0]
    clockwise_vertices.append((x_pre, y_pre))

    for x, y in vertices[1:]:
        if y >= y_pre:
          clockwise_vertices.append((x, y))


if __name__ == '__main__':
    mask_path = sys.argv[1]
    mask_num = sys.argv[2]
    vertices = get_vertices(mask_path, 5)

    unzip_vertices = ([i for i, j in vertices], [j for i, j in vertices])
    mask = plt.imread(mask_path)        ## load the image mask
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    area = np.where(mask == 1) ## indices of the pixels involved in mask
    mask[area] = 0

    mask = np.rot90(np.rot90(np.rot90(mask)))
    mask[unzip_vertices] = 1
    plt.imsave('vertices.png', mask)
    
    print(vertices)
    # print(v)
    pprint(get_vertices_by_m(mask_path, t=0.5))

    with open('strokes'+str(mask_num)+'.bezier', 'w') as f:
        for vertex in vertices:
            f.write(str(vertex[0])+'\t'+str(vertex[1])+'\n')




##   python exp/inference/inference.py  --loadmodel /home/saket/Graphonomy/data/pretrained_model/deeplab_v3plus_v3.pth --img_path ./img/messi.jpg --output_path ./img/ --output_name test





