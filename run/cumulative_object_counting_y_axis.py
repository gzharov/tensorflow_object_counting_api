# Object detection imports
from utils2 import backbone
from api import object_counting_api

#input_video = "cumulative.mp4"
#input_video = "Запись улицы 2.1.mp4"
input_video = "./test_videos/3.mp4"

#detection_graph, category_index = backbone.set_model('faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28', 'mscoco_label_map.pbtxt')

detection_graph, category_index = backbone.set_model('my_classifier', 'my_labelmap.pbtxt')

is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
roi = 700 # roi line position
deviation = 6 # the constant that represents the object counting area

object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation) # counting all the objects
