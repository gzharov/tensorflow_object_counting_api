import tensorflow as tf
import csv
import cv2
import numpy as np
#from utils2 import visualization_utils as vis_util
from sort import *

import collections
import functools
import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import numpy
import os




#-----------------
#drawing for task
def draw_bounding_box_on_image_array_for_parked_cars(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     ):

  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image_for_parked_cars(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list)
  np.copyto(image, np.array(image_pil))
  
  
  
def draw_bounding_box_on_image_for_parked_cars(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=()
                               ):

  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  #print('im_width, im_height = ',im_width, im_height)
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

#-----------------


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='lime',
                                     thickness=4,
                                     display_str_list=(),
                                     ):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list)
  np.copyto(image, np.array(image_pil))
  
  
  
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='lime',
                               thickness=4,
                               display_str_list=()
                               ):
  """Adds a bounding box to an image.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  #print('im_width, im_height = ',im_width, im_height)
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    #print('lfrb = ',left, right, top, bottom)
  
  try:
    font = ImageFont.truetype('arial.ttf', 16)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_list[0] = display_str_list[0]
  #csv_line = str (predicted_direction) # csv line created
  
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    #print('text_width, text_height, margin = ',text_width, text_height, margin)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
    
    
    
### Part eval



def iou_between_two_boxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[3] - boxA[1]) * (boxA[2] - boxA[0]))
    boxBArea = abs((boxB[3] - boxB[1]) * (boxB[2] - boxB[0]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou  
  
 
def intersection_track_arrays(track_start,track_end):
    new_start = []
    new_end = []
    track_end_copy = np.copy(track_end)
    for i in range(len(track_start)):
        #if(len(track_end_copy)==0):
            #break
        for j in range(len(track_end_copy)):
            if (track_start[i][4]==track_end_copy[j][4]):
                new_start.append(track_start[i].tolist())
                new_end.append(track_end_copy[j].tolist())
                track_end_copy = np.delete(track_end_copy,j, axis = 0)
                break
    
    return new_start, new_end
        
  
  
  
def object_tracking(input_video, detection_graph, category_index, min_score_thresh=0.5):

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
        
        #create tracker
        mot_tracker = Sort()
        mot_tracker.__init__(max_age = fps)
        
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
          
            #count_frames = 0
            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame
                #count_frames +=1
                
                    
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                    
                    
                track_bboxes = np.reshape(boxes, (len(boxes[0]), 4))
                track_scores = np.reshape(scores,(len(boxes[0])))
                #boxes_update, track_id = track(track_bboxes, track_scores)
                track_bboxes_update = [track_bboxes[i].tolist() for i in range(len(track_bboxes)) if track_scores[i]>=min_score_thresh]
              
                #print(track_bboxes_update)
                #break
                if(len(track_bboxes_update)==0):
                    track_bbs_ids = mot_tracker.update()
                else:
                    track_bbs_ids = mot_tracker.update(np.array(track_bboxes_update))
                #track_bbs_ids = mot_tracker.update(np.array(track_bboxes_update))
                print(len(track_bbs_ids),len(track_bboxes_update))
  
                track_id = []
                track_bboxes_update2 = []
                #for i in range(len(track_bbs_ids)):
                for i in range(len(track_bbs_ids)-1,-1,-1):
                    #track_id.append(int(track_bbs_ids[i][4]))
                    track_id.append(int(track_bbs_ids[i][4]))
                    track_bboxes_update2.append(track_bbs_ids[i][:4].tolist())
                
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                #print(track_bboxes_update, track_id)
                #break
            
            
                # Visualization of the results of a detection. 
                #im_width, im_height = input_frame.size
                #print(width, height)
                
                for i in range(len(track_bboxes_update2)):
                    #print("len =",len(track_bboxes_update), len(track_id),i)
                    ymin,xmin,ymax,xmax = track_bboxes_update2[i][0], track_bboxes_update2[i][1], track_bboxes_update2[i][2], track_bboxes_update2[i][3]
                    #display_str_list = ["ID: "+str(track_id[i])]
                    dsl = ["ID: "+str(track_id[i])]
                    draw_bounding_box_on_image_array(input_frame,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     display_str_list = dsl)
                    
                    
                    
                #cv2.imwrite('output_image.jpg', input_frame)
                #break
                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()





### Part for not_real_time


def intersection_bboxes_arrays(arr1,arr2):
    
    buff = []
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            if (iou_between_two_boxes(arr1[i], arr2[j])>=0.8):
                buff.append(arr1[i])
                break
    
    if len(buff)==0:
        return []
    else:
        return buff
    
            
            
def find_box_in_array(cur_mark_buffer, box):
    
    for i in range(len(cur_mark_buffer)):
        if iou_between_two_boxes(cur_mark_buffer[i], box)>0.8:
            return True
    return False
    
    
    
    
def update_mark_buffer(cur_mark_buffer, cur_track_start, cur_track_end):
    
    if len(cur_mark_buffer)==0:
        cur_mark_buffer = []
        new_start, new_end = intersection_track_arrays(cur_track_start,cur_track_end)
        for i in range(len(new_start)):
            iou = iou_between_two_boxes(new_start[i],new_end[i])
            if (iou >= 0.8):
                cur_mark_buffer.append(new_end[i])
    else:
        cur_mark_buffer_copy = list(cur_mark_buffer)
        cur_mark_buffer = []
        new_start, new_end = intersection_track_arrays(cur_track_start,cur_track_end)
        
        if len(new_end)==0:
            cur_mark_buffer = list(intersection_bboxes_arrays(cur_track_end,cur_mark_buffer_copy))
        else:
            new_buffer = list(intersection_bboxes_arrays(cur_track_end, cur_mark_buffer_copy))
            second_new_buffer = list(intersection_bboxes_arrays(new_start, new_end))
            new_buffer_all = list(new_buffer)
            for i in range(len(second_new_buffer)):
                if (find_box_in_array(new_buffer, second_new_buffer[i])==False):
                    new_buffer_all.append(second_new_buffer[i])
            cur_mark_buffer = list(new_buffer_all)    
                    
    if len(cur_mark_buffer)==0:
        return []
    else:
        return cur_mark_buffer
            
        
        
        
def get_results_for_not_real_time(cur_mark_array, cur_mark_buffer):
    
    array_to_draw = []
    #print("cur_mark_buffer = ",cur_mark_buffer)
    #print("cur_mark_array = ",cur_mark_array)
    if len(cur_mark_buffer)!=0:
        for i in range(len(cur_mark_array)):
            if len(cur_mark_array[i])>0:
                buff = intersection_bboxes_arrays(cur_mark_array[i], cur_mark_buffer)
                array_to_draw.append(buff)
            else:
                array_to_draw.append([])
    else:
        for i in range(len(cur_mark_array)):
            array_to_draw.append([])
               
    return array_to_draw
    



def parked_cars_not_realtime(input_video, detection_graph, category_index, interval_seconds, min_score_thresh=0.5):

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        counter_fps_end = interval_seconds*fps

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
        
        #create tracker
        mot_tracker = Sort()
        
        #create frame mark arrays
        mark_array = [] # store bboxes 
        mark_buffer = list([])# store detections of one interval
        detections_to_draw = []# store all detections to drawing
        
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
          
            count_frames = 0
            counter_parked_cars = 0
            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                
                print("first stage")
                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame
                count_frames +=1
                
                    
                #print("mark_buffer = ", mark_buffer)
                #print("mark_array = ", mark_array)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                    
                    
                track_bboxes = np.reshape(boxes, (len(boxes[0]), 4))
                track_scores = np.reshape(scores,(len(boxes[0])))
                #boxes_update, track_id = track(track_bboxes, track_scores)
                track_bboxes_update = [track_bboxes[i].tolist() for i in range(len(track_bboxes)) if track_scores[i]>=min_score_thresh]
              
                #print(track_bboxes_update)
                #break
                
                if (count_frames == 1):
                    
                    if(len(track_bboxes_update)==0):
                        track_bbs_ids = mot_tracker.update()
                        mark_array.append([])
                    else:
                        track_bbs_ids = mot_tracker.update(np.array(track_bboxes_update))
                        mark_array.append(track_bbs_ids.tolist())
                    
                    track_start = track_bbs_ids
                    
                elif (count_frames == counter_fps_end):
                    
                    count_frames = 0
                    
                    if(len(track_bboxes_update)==0):
                        track_bbs_ids = mot_tracker.update()
                        mark_array.append([])
                    else:
                        track_bbs_ids = mot_tracker.update(np.array(track_bboxes_update))
                        mark_array.append(track_bbs_ids.tolist())
                    
                    track_end = track_bbs_ids
                    
                    if (len(track_start)==0 or len(track_end)==0):
                        mark_array = []
                        mark_buffer = []
                    else:
                        if len(mark_buffer)==0:
                            mark_buffer = update_mark_buffer([], track_start, track_end)
                        else:
                            mark_buffer = update_mark_buffer(mark_buffer, track_start, track_end)

                        #update_detections_to_draw = get_results_for_not_real_time(mark_array, mark_buffer)
                        if len(mark_buffer)==0:
                            update_detections_to_draw = get_results_for_not_real_time(mark_array, [])
                        else:
                            update_detections_to_draw = get_results_for_not_real_time(mark_array, mark_buffer)
                        
                        for i in range(len(update_detections_to_draw)):
                            detections_to_draw.append(update_detections_to_draw[i])
                            
                        mark_array = []
                    
                else:
                    
                    if(len(track_bboxes_update)==0):
                        track_bbs_ids = mot_tracker.update()
                        mark_array.append([])
                    else:
                        track_bbs_ids = mot_tracker.update(np.array(track_bboxes_update))
                        mark_array.append(track_bbs_ids.tolist())
                  
                    

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()
            
            #Draw results
            
            cap = cv2.VideoCapture(input_video)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            counter_fps_end = interval_seconds*fps

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
            
            draw_index = 0
            
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame
                
                if draw_index<=(len(detections_to_draw) - 1):
                    if len(detections_to_draw[draw_index])>0:
                        for j in range(len(detections_to_draw[draw_index])):
                            ymin,xmin,ymax,xmax = detections_to_draw[draw_index][j][0], detections_to_draw[draw_index][j][1], detections_to_draw[draw_index][j][2], detections_to_draw[draw_index][j][3]
                            dsl = []
                            draw_bounding_box_on_image_array_for_parked_cars(input_frame,
                                        ymin,
                                        xmin,
                                        ymax,
                                        xmax,
                                        color='blue',
                                        thickness=4,
                                        display_str_list=dsl)
                                            # insert information text to video frame
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        'Parked cars: ' + str(len(detections_to_draw[draw_index])),
                        (10, 35),
                        font,
                        0.8,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        )
                
                                         
                    output_movie.write(input_frame)
                    print ("writing frame")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break 
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        'Parked cars: ' + str(0),
                        (10, 35),
                        font,
                        0.8,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        )
                                
                    output_movie.write(input_frame)
                    print ("writing frame")
                    
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break 
                    
                draw_index +=1
            
            
