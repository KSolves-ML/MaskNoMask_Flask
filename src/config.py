from tensorflow.keras.models import load_model
import cv2
#caffeModel = "/Users/gauravsrivastava/Desktop/KSOLVE/Face-detection-with-OpenCV-and-deep-learning-master/models/res10_300x300_ssd_iter_140000.caffemodel"
caffeModel = "res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "deploy.prototxt.txt"
maskNet = load_model('mask_detector.model')
faceNet = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)



