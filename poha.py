'''
Rotoscopy helping function
'''
from sklearn.metrics import confusion_matrix  
import numpy as np


import numpy as np
import glob
import argparse
import cv2
import os
import sys
from matplotlib import pyplot as plt

def compute_iou(Y_pred, Y_true, labels):
    '''
    Computes mean IOU given the true mask and the predicted mask.

    Arguments ::
        Y_pred --
        Y_true --
        labels --
    Return ::
        meanIOU --
    '''
    # ytrue, ypred is a flatten vector
    Y_pred = Y_pred.flatten()
    Y_true = Y_true.flatten()
    
    current = confusion_matrix(Y_true, Y_pred, labels)
    
    intersection = np.diag(current)

    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def get_individual_mask(pre_mask_path, dir_name='', k=100):
    '''
    Generates individual masks.

    Arguments ::
        pre_mask_path -- str | path where the mask is stored
        dir_name -- optional | str | directory name of the masks
        k -- int | optional | mask descriptor
    Return ::
        k -- int | descriptor of the last mask
    '''
    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    pre_mask = cv2.imread(pre_mask_path, 0)
    individual_masks = []
    row_count = 0
    col_count = 0
    flag = True
    pre_row = 0
    pre_col = 0
    init_k = k
    to_del = []

    if len(pre_mask.shape) > 2:
        pre_mask = np.squeeze( cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY))
    
    for row in range(pre_mask.shape[0]):
        if np.sum(pre_mask[row]) > 0 and flag:
            row_count += 1
            flag = False
        elif np.sum(pre_mask[row]) == 0 and not flag:
            flag = True
            if row_count > 0 and row > 0:
                new_mask = np.zeros_like(pre_mask)
                new_mask[pre_row : row] = pre_mask[pre_row : row]
                if np.sum(new_mask == pre_mask) != np.sum(pre_mask == pre_mask):
                    to_del.append(dir_name+'mask'+str(k-1)+'.png')
                cv2.imwrite(dir_name+'mask'+str(k-1)+'.png', new_mask)
                individual_masks.append(dir_name+'mask'+str(k-1)+'.png')
                k += 1
                pre_row = row

    flag = True
    if k - init_k == 0:
        individual_masks.append(pre_mask_path)
    print(k, individual_masks)

    for row_sep_mask in individual_masks:
        pre_mask = cv2.imread(row_sep_mask, 0)
        temp_masks = []
        flag = True
        if len(pre_mask.shape) > 2:
            pre_mask = np.squeeze( cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY))

        for col in range(pre_mask.shape[1]):
            if np.sum(pre_mask[:, col]) > 0 and flag:
                col_count += 1
                flag = False
            elif np.sum(pre_mask[:, col]) == 0 and not flag:
                flag = True
                if col_count > 0 and col > 0:
                    new_mask = np.zeros_like(pre_mask)
                    new_mask[:, pre_col : col] = pre_mask[:, pre_col : col]
                    if np.sum(new_mask == pre_mask) != np.sum(pre_mask == pre_mask):
                        to_del.append(dir_name+'mask'+str(k)+'.png')
                    cv2.imwrite(dir_name+'mask'+str(k)+'.png', new_mask)
                    temp_masks.append(dir_name+'mask'+str(k)+'.png')
                    k += 1
                    pre_col = col
        if k > init_k:
            init_k = k
        if len(temp_masks) > 0:
            individual_masks.remove(row_sep_mask)
            individual_masks.extend(temp_masks)

    flag = True
    if k > init_k:
        k -= 1
        init_k = k
    print(k, individual_masks)

    for row_sep_mask in individual_masks:
        pre_mask = cv2.imread(row_sep_mask, 0)
        temp_masks = []
        flag = True

        if len(pre_mask.shape) > 2:
            pre_mask = np.squeeze( cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY))

        for row in range(pre_mask.shape[0]):
            if np.sum(pre_mask[row]) > 0 and flag:
                row_count += 1
                flag = False
            elif np.sum(pre_mask[row]) == 0 and not flag:
                flag = True
                if row_count > 0 and row > 0:
                    new_mask = np.zeros_like(pre_mask)
                    new_mask[pre_row : row] = pre_mask[pre_row : row]
                    if np.sum(new_mask == pre_mask) != np.sum(pre_mask == pre_mask):
                        to_del.append(dir_name+'mask'+str(k)+'.png')
                    cv2.imwrite(dir_name+'mask'+str(k)+'.png', new_mask)
                    k += 1
                    pre_row = row

    return k

def remove_same_from_list(dir_name, rm_list):
    '''
    Removes similar images from the rm_list and the list
    
    Arguments ::
        dir_name -- str | directory name, where the masks are saved.
        rm_list -- list | contains image names to be removed and similar images to be removed
    '''

    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    for image_name in os.listdir(dir_name):
        if image_name.split('.')[-1] == 'png':
            if image_name in rm_list:
                continue
            else:
                image = cv2.imread(dir_name + image_name, 0)
                check = np.sum(image == image)
                if np.max(image) == 255:
                    image = image / 255
                for rm_image_name in rm_list:
                    rm_image = cv2.imread(rm_image_name, 0)
                    if np.max(rm_image) == 255:
                        rm_image = rm_image / 255
                    if np.sum(image == rm_image) == check:
                        os.remove(dir_name + image_name)
                        break

def remove_from_list(dir_name, rm_list):
    '''
    Removes images from rm_list inside dir
    
    Arguments ::
        dir_name -- str | directory name, where the masks are saved.
        rm_list -- list | contains image names to be removed and similar images to be removed
    '''

    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    for image_name in os.listdir(dir_name):
        if image_name.split('.')[-1] == 'png':
            if image_name in rm_list:
                continue
                os.remove(dir_name + image_name)

def remove_empty(dir_name):
    '''
    Given a directory name, removes all the empty images.

    Arguments ::
        dir_name -- str | directory name, where the masks are saved.
    '''
    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    for image_name in os.listdir(dir_name):
        if image_name.split('.')[-1] == 'png':
            if np.sum(cv2.imread(dir_name + image_name, 0)) == 0:
                os.remove(dir_name + image_name)
    
def remove_same(dir_name):
    '''
    Given a directory name, removes all the images which are same.

    Arguments ::
        dir_name -- str | directory name, where the masks are saved.
    '''
    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    for image_name in os.listdir(dir_name):
        for image2_name in os.listdir(dir_name):
            if image_name != image2_name:
                if image_name.split('.')[-1] == 'png':
                    if np.sum(cv2.imread(dir_name + image_name, 0) != cv2.imread(dir_name + image2_name, 0)) == 0:
                        os.remove(dir_name + image_name)


def remove_low_area_mask(dir_name):
    '''
    Given a directory name, removes all the masks for which the area is very low.

    Arguments ::
        dir_name -- str | directory name, where the masks are saved.
    '''
    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    for image_name in os.listdir(dir_name):
        if image_name.split('.')[-1] == 'png':
            image = cv2.imread(dir_name + image_name, 0)
            max_pix = np.max(image)
            if np.sum(image) < 18 * max_pix:
                    os.remove(dir_name + image_name)

def remove_superset(dir_name):
    '''
    Given a directory name, removes all the images which are superset of any image.

    Arguments ::
        dir_name -- str | directory name, where the masks are saved.
    '''
    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    for image_name in os.listdir(dir_name):
        if not os.path.isfile(dir_name + image_name):
            continue
        for image2_name in os.listdir(dir_name):
            if not os.path.isfile(dir_name + image_name):
                break
            if not os.path.isfile(dir_name + image2_name):
                continue
            if image_name != image2_name:
                if image_name.split('.')[-1] == 'png':
                    image = cv2.imread(dir_name + image_name, 0)
                    image2 = cv2.imread(dir_name + image2_name, 0)

                    max_pix = np.max(image)
                    image_bool = image == max_pix
                    image2_bool = image == max_pix
                    imagex_bool = np.logical_and(image_bool, image2_bool)
                    # print(image_name, image2_name, np.sum(imagex_bool))
                    
                    if np.sum(image_bool != imagex_bool) == 0:
                        os.remove(dir_name + image_name)
                    elif np.sum(image2_bool != imagex_bool) == 0:
                        os.remove(dir_name + image2_name)
                    if not os.path.isfile(dir_name + image_name):
                        break


def get_labels(mask):
    """
    Gets all the lables in a RGB segmented image

    Arguments ::
        mask -- ndarray | ndarray representing mask
    Return ::
        n_labels -- ndarray | representing unique labels without background
    """
    labels = np.unique(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    return labels[np.where(labels != 0)]

def isolate_class(mask, class_id):
    """
    Extracts greyscale masks of all the classes given a RGB mask

    Arguments ::
        
    Returns ::
        
    """
    gmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    temp_mask = np.zeros_like(gmask)
    temp_mask[gmask == class_id] = 255

    # temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_BGR2GRAY)
    # temp_mask[temp_mask > 0] = 255

    return temp_mask

if __name__ == '__main__':
    mask = cv2.imread(sys.argv[1])
    cv2.imwrite('a.png', cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    labels = get_labels(mask)
    print(labels)

    i = 1
    for label in labels:
        label_mask = isolate_class(mask, label)
        cv2.imwrite('mask'+str(i)+'.png', label_mask)
        i += 1

    '''
    dir_name = sys.argv[1]
    if dir_name[-1] != '/':
        dir_name = dir_name+'/'

    k = 200
    for image_name in os.listdir(dir_name):
        if image_name.split('.')[-1] == 'png':
            k = get_individual_mask(dir_name + image_name, dir_name, k=k)

    remove_empty(dir_name)
    remove_same(dir_name)
    '''
