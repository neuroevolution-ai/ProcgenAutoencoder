import glob

import cv2 as cv


method = cv.TM_SQDIFF_NORMED

# Path  / Threshold # Thresholds were set manually, by careful examination of examples
template_blue_locks = ([img for img in glob.glob("./templates_heist/blue_lock/*.png")],0.05)
template_blue_keys  = ([img for img in glob.glob("./templates_heist/blue_key/*.png")],0.07)
template_green_locks = ([img for img in glob.glob("./templates_heist/green_lock/*.png")],0.06)
template_green_keys  = ([img for img in glob.glob("./templates_heist/green_key/*.png")],0.15)
template_red_locks = ([img for img in glob.glob("./templates_heist/red_lock/*.png")],0.07)
template_red_keys  = ([img for img in glob.glob("./templates_heist/red_key/*.png")],0.22)
template_goal = ([img for img in glob.glob("./templates_heist/goal/*.png")],0.05)


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





def find_template(img, templates, threshold):
    '''
    For a given image finds the best matching template. If no good template is found this function returns False
    otherwise True
    :param img: The image that is to be examined
    :param templates: List of Templates locations
    :param threshold: Threshold that is used for the templates. If the error is bigger than the given treshhold
            match will not be accepted
    :return: returns True if one good template match is found, False Otherwise. Returns the location of the best
             Matching Template in the picture for further use.
    '''
    best_matching =threshold
    best_min,best_max,best_minloc, best_maxloc = 0,0,0,0
    for next_template in templates:
        next_template = cv.imread(next_template,cv.COLOR_BGR2RGB)
        w,h,_ = next_template.shape[::-1]
        result = cv.matchTemplate(img, next_template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if min_val < best_matching:
            best_min =min_val
            best_max =max_val
            best_minloc =min_loc
            best_maxloc =max_loc
            best_matching = min_val
    if best_matching == threshold:
        return False, [img, best_min, best_max, best_minloc, best_maxloc, w, h]

    return True , [img, best_min, best_max, best_minloc, best_maxloc, w, h]





