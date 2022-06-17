from utils2 import backbone
import track_module

#input_video = "test_videos/3.mp4"
#input_video = "test_videos/sub-1504619634606.mp4"
#input_video = "test_videos/2.1.mp4"
input_video = "test_videos/4k.1.mp4"


detection_graph, category_index = backbone.set_model('my_classifier', 'my_labelmap.pbtxt')

track_module.parked_cars_not_realtime(input_video, detection_graph, category_index, 20, min_score_thresh=0.1)