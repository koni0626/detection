# coding:UTF-8

import glob
import numpy as np
import train
import cv2
import conf
import os


if __name__ == '__main__':
    model = train.my_yolov1()
    model.load_weights("../weight/yolov1.40-0.00-0.85.hdf5")
    file_list = glob.glob(os.path.join(conf.TRAIN_DATA_PATH, "*.jpg"))
    
    for p, file_name in enumerate(file_list):
        original_img = cv2.imread(file_name)
        img = cv2.resize(original_img, conf.IMAGE_SIZE).T/255.
        test_x = np.array([img])
        result = model.predict(test_x)
        record = result[0]
        size = len(record)
        print(record)
        for i in range(0, size, conf.CLASS_NUM*5):
            for c in range(conf.CLASS_NUM):
                index = i + c * 5
                if record[index] > 0.5:
                    #cがクラスを表す
                    print(record[index:index+5])
                    #print("pos="+str(i))
                    o_h, o_w = original_img.shape[0:2]
                    c_x = record[index + 1] * o_w
                    c_y = record[index + 2] * o_h
                    w = record[index + 3] * o_w
                    h = record[index + 4] * o_h
                    left = c_x - w/2.
                    top = c_y - h/2.
                    cv2.rectangle(original_img, (int(left), int(top)), (int(left + w), int(top + h)), (0, 255, 0), thickness=2)
        outputfile = "{}.jpg".format(p)
        cv2.imwrite(outputfile, original_img)
            
        
    
