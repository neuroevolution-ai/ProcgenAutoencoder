import glob

import cv2 as cv
from matplotlib import pyplot as plt


method = cv.TM_SQDIFF_NORMED

filenames = [img for img in glob.glob("research/*.png")]

# Path  / Threshold
template_blue_locks = ([img for img in glob.glob("research/templates/blue_lock/*.png")],0.05)
template_blue_keys  = ([img for img in glob.glob("research/templates/blue_key/*.png")],0.07)
template_green_locks = ([img for img in glob.glob("research/templates/green_lock/*.png")],0.06)
template_green_keys  = ([img for img in glob.glob("research/templates/green_key/*.png")],0.15)
template_red_locks = ([img for img in glob.glob("research/templates/red_lock/*.png")],0.07)
template_red_keys  = ([img for img in glob.glob("research/templates/red_key/*.png")],0.22)
template_goal = ([img for img in glob.glob("research/templates/goal/*.png")],0.05)


def has_blue_locks(img):
    has , _= find_template(img,template_blue_locks[0],template_blue_locks[1])
    return has

def has_green_locks(img):
    has, _ =find_template(img,template_green_locks[0],template_green_locks[1])
    return has

def has_red_locks(img):
    has , _ = find_template(img, template_red_locks[0],template_red_locks[1])
    return has

def has_blue_keys(img):
    has , _= find_template(img,template_blue_keys[0],template_blue_keys[1])
    return has

def has_green_keys(img):
    has, _ =find_template(img,template_green_keys[0],template_green_keys[1])
    return has

def has_red_keys(img):
    has , _ = find_template(img, template_red_keys[0],template_red_keys[1])
    return has

def has_no_Keys_and_Doors(img):
    return not has_blue_locks(img) and not has_green_locks(img) and not has_red_locks(img) and not has_blue_keys(img)  and not has_green_keys(img) and not has_red_keys(img)


def has_goal(img):
    has , _ = find_template(img, template_goal[0], template_goal[1])
    return has


templates =[]
templates.append(template_red_locks)





def find_template(img,templates,threshhold):
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    best_matching =threshhold
    best_min,best_max,best_minloc, best_maxloc = 0,0,0,0
    for next_template in templates:
        next_template = cv.imread(next_template,cv.COLOR_BGR2RGB)
        w,h,_ = next_template.shape[::-1]
        result = cv.matchTemplate(img, next_template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if min_val < best_matching:
            bigger_template = cv.resize(next_template, (0, 0), fx=5, fy=5, interpolation=cv.INTER_AREA)
            best_min =min_val
            best_max =max_val
            best_minloc =min_loc
            best_maxloc =max_loc
            best_matching = min_val
            cv.imshow("Template", bigger_template)
    if best_matching == threshhold:
        return False, [img, best_min, best_max, best_minloc, best_maxloc, w, h]

    return True , [img, best_min, best_max, best_minloc, best_maxloc, w, h]


if __name__ == '__main__':
    images = []
    for img in filenames:
        img = cv.imread(img)
        for template in templates:
            # Apply template Matching
            found, [img, min_val, max_val, min_loc, max_loc,w,h] = find_template(img, template[0], template[1])
            img2 = img.copy()
            if found:
                top_left = min_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv.rectangle(img2, top_left, bottom_right, (0, 0, 255), 2)
                cv.waitKey(0)
            bigger = cv.resize(img2, (0, 0), fx=5, fy=5, interpolation=cv.INTER_AREA)
            bigger_old = cv.resize(img, (0, 0), fx=5, fy=5, interpolation=cv.INTER_AREA)

            cv.imshow("Gefunden", bigger)
            cv.imshow("Altes Bild", bigger_old)
            cv.waitKey(0)




