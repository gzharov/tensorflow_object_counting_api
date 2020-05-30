from utils2 import backbone
from api import object_counting_api

#input_video = "Запись3.mp4"
input_video = "cumulative.mp4"

#indicate the name of the directory where frozen_inference_graph.pb is located and the label map in data folder
detection_graph, category_index = backbone.set_model('faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28', 'mscoco_label_map.pbtxt')

targeted_objects = "car, bus" # (for counting targeted objects) change it with your targeted objects

is_color_recognition_enabled = 0

object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects)