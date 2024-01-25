import os
import cv2
import numpy as np

# Specify the .dcm folder path
folder_path = "path"

# Specify the output jpg folder path
jpg_folder_path = "path"

images_path = os.listdir(folder_path)

def check_color(img):
    """
    Check the color image for is black or white

    Args
        img: Breast mammogram image 

    Returns
        white/black: If image has more white places return white, otherwise black 
    
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    pixels = cv2.countNonZero(thresh)

    ratio = (pixels/(h * w)) * 100
    if ratio < 50:
        return "white"
    else:
        return "black"

def crop_breast(img, path:str, name:str):
    """
    Crop the empty place of breast mammogram

    Args
        img: Breast mammogram image
        path: Path for saving cropped image
        name: Cropped image name

    """
    color = check_color(img)
    if color == "black":
        img = np.invert(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    temp_path = os.path.join(path, name)
    cv2.imwrite(str(temp_path), dst)

def main():
    for _, x in enumerate(images_path):
        p = folder_path + "\\" + str(x)
        p1 = os.listdir(p)
        new_path = jpg_folder_path + "\\" + str(x)
        if len(p1) != 0:
            for _ , image in enumerate(p1):
                try:
                    im = cv2.imread(os.path.join(p, image))
                    crop_breast(im, new_path, image)
                    print("done: ", p)
                except Exception as err:
                    print(err)
                    continue

if __name__ == "__main__":
    main()