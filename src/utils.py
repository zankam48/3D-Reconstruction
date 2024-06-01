import cv2

path = r'D:\\Coding Project\\3d reconstruction\\images\\'
img_len = 37


def img_path(path, img_nth):
    img_list = []
    for i in range(0, img_len):
        if (i//10) < 1:
            img_file = "viff.00{}".format(i) + ".jpg"
        else:
            img_file = "viff.0{}".format(i) + ".jpg"
        img_file = path + img_file
        img_list.append(img_file)
    return img_list[img_nth - 1]
