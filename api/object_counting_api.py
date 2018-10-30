# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import time
import datetime
import numpy as np
from scipy import stats

# Variables
total_passed_vehicle = 0  # using it to count vehicles

def cumulative_object_counting_x_axis_bps(model_name, input_video, detection_graph, category_index, is_color_recognition_enabled, fps,
                                      width, height, roi):
    total_passed_vehicle = 0

    # initialize .csv
    with open(input_video.strip('.mp4') + '_' + model_name + '_INOUT' + '_measure.txt', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter( input_video.strip('.mp4') + '_' + model_name + '_INOUT' + '_output.mp4', fourcc, fps, (width, height))

    # input video
    cap = cv2.VideoCapture(input_video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Numero de Frames:" , num_frames, "FPS:", fps, "Tempo do video (s):", num_frames*fps)

    total_passed_vehicle = 0
    total_in = 0
    total_out = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
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

            inf_media=[]
            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                inf_start = time.time()
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                inf = (time.time() - inf_start)
                inf_media.append(inf)
                moda = stats.mode(inf_media)
                #if cap.get(1) % 50 == 0 :
                #     print("Inference elapsed time (s) %1.4f" % inf, "Moda:", moda[0])
                
				# insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                vis_start = time.time()
                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
                                                                                                             targeted_objects="person",
                                                                                                             x_reference=roi,
                                                                                                             deviation=1.24,  #variação sobre o roi
                                                                                                             min_score_thresh=.8,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                vis = time.time() - vis_start				
                #size, direction = csv_line.split(',')
				#print("Visualization elapsed time (s) %1.4f" % vis)
                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
                    if(csv_line != "not_available"):
                        size, direction = csv_line.split(',')
                        if(direction =="up" ):
                            total_passed_vehicle = total_passed_vehicle + counter
                            total_out = total_out + counter 
                            #print("Inference elapsed time (s) %1.4f" % inf, "Moda:", moda[0])
                        if(direction =="down"):
                            total_passed_vehicle = total_passed_vehicle - counter
                            total_in = total_in + counter
                            #print("Inference elapsed time (s) %1.4f" % inf, "Moda:", moda[0])
                else:
                    cv2.line(input_frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

                #total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    #'Saldo: ' + str(total_passed_vehicle) + ' Entradas: ' + str(total_in) + ' Saidas: ' + str(total_out),
                    ' Entradas: ' + str(total_in) + ' Saidas: ' + str(total_out),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                #cv2.putText(
                #    input_frame,
                #    'ROI Line',
                #    (545, roi - 10),
                #    font,
                #    0.6,
                #    (0, 0, 0xFF),
                #    2,
                #    cv2.LINE_AA,
                #)
                rec_start = time.time()
                output_movie.write(input_frame)
                rec = time.time() - rec_start
                #print("Recording elapsed time (s) %1.4f" % rec)
                #print("csv_line-->",csv_line)
                #print ("writing frame", str(total_passed_vehicle))
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if(csv_line != "not_available"):
                    with open(input_video.strip('.mp4') + '_' + model_name + '_INOUT' + '_measure.txt', 'a') as f:
                        writer = csv.writer(f)
                        size, direction = csv_line.split(',')
                        csv_line = csv_line + ',' + str(int(cap.get(1))) + ',' + str(total_in) + ',' + str(total_out)
                        print("csv_line:",csv_line )
                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()



def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps,
                                      width, height, roi):
    total_passed_vehicle = 0

    # initialize .csv
    with open('object_counting_report.csv', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter( '_output.avi', fourcc, fps, (width, height))

    # input video
    cap = cv2.VideoCapture(input_video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Numero de Frames:" , num_frames, "FPS:", fps, "Tempo do video (s):", num_frames*fps)

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
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

            inf_media=[]
            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                inf_start = time.time()
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                inf = (time.time() - inf_start)
                inf_media.append(inf)
                moda = stats.mode(inf_media)
                print("Inference elapsed time (s) %1.4f" % inf, "Moda:", moda[0])
                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                vis_start = time.time()
                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
																											 #targeted_objects="person",
                                                                                                             x_reference=roi,
                                                                                                             deviation=1,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                vis = time.time() - vis_start
                #print("Visualization elapsed time (s) %1.4f" % vis)
                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detectado: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi - 10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )
                rec_start = time.time()
                output_movie.write(input_frame)
                rec = time.time() - rec_start
                #print("Recording elapsed time (s) %1.4f" % rec)
                #print("csv_line-->",csv_line)
                #print ("writing frame", str(total_passed_vehicle))
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if(csv_line != "not_available"):
                    with open('traffic_measurement.csv', 'a') as f:
                            writer = csv.writer(f)
                            size, direction = csv_line.split(',')
                            writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()


def cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps,
                                      width, height, roi):
    total_passed_vehicle = 0

    # initialize .csv
    with open('object_counting_report.txt', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_output.mp4', fourcc, fps, (width, height))

    # input video
    cap = cv2.VideoCapture(input_video)

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
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

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
                                                                                                             y_reference=roi,
                                                                                                             deviation=7,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected : ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi - 10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )

                output_movie.write(input_frame)
                #print("writing frame")
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.txt', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):
    # initialize .csv
    with open('object_counting_report.txt', 'w') as f:
        writer = csv.writer(f)
        csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"
        writer.writerows([csv_line.split(',')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('the_output.mp4', fourcc, fps, (width, height))

    # input video
    cap = cv2.VideoCapture(input_video)

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    height = 0
    width = 0
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

            inf_start=[]
            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(
                                                                                                          classes).astype(
                                                                                                          np.int32),
                                                                                                      np.squeeze(
                                                                                                          scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if (len(counting_mode) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0, 255, 255), 2,
                                cv2.FONT_HERSHEY_SIMPLEX)

                output_movie.write(input_frame)
                #print("writing frame")
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if (csv_line != "not_available"):
                    with open('traffic_measurement.txt', 'a') as f:
                        writer = csv.writer(f)
                        size, direction = csv_line.split(',')
                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()


def targeted_object_counting(model_name,input_video, detection_graph, category_index, is_color_recognition_enabled,
                             targeted_object, fps, width, height):
    # initialize .csv
    with open(input_video.strip('.mp4') + '_' + model_name + '_QTD' + '_measure.txt', 'w') as f:
        writer = csv.writer(f)
        csv_line = "ID, Data, Tempo, Visitantes"
        writer.writerows([csv_line.split(',')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(input_video.strip('.mp4') + '_' + model_name + '_QTD' + '_output.mp4', fourcc, fps, (width, height))
    # input video
    cap = cv2.VideoCapture(input_video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Numero de Frames:" , num_frames, "FPS:", fps, "Tempo do video (s):", num_frames*fps)

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    the_result = "..."
    width_heigh_taken = True
    height = 0
    width = 0
    intervalo_visitantes = []
    antes =0
    agora = int(time.time())
    intervalo = 59 # intervalo em segundos + 1
    id=1
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

            inf_media=[]
            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                inf_start = time.time()
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                inf = (time.time() - inf_start)
                #inf_media.append(inf)
                #moda = stats.mode(inf_media)
                #print("Inference elapsed time (s) %1.4f" % inf, "Moda:", moda[0])

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                   input_frame,
                                                                                                   1,
                                                                                                   is_color_recognition_enabled,
                                                                                                   np.squeeze(boxes),
                                                                                                   np.squeeze(
                                                                                                       classes).astype(
                                                                                                       np.int32),
                                                                                                   np.squeeze(scores),
                                                                                                   category_index,
                                                                                                   targeted_objects=targeted_object,
                                                                                                   use_normalized_coordinates=True,
                                                                                                   line_thickness=4)

                visitantes = 0
                if (len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    #print( "the_result[10:]:" , the_result[10:])
                    visitantes = int(the_result[10:12])
                    inf_media.append(visitantes)
                    moda = np.mean(inf_media)
                    moda = np.around(moda,decimals=2)
                    cv2.putText(input_frame, "visitantes: " + str(visitantes) , (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)

                with open(input_video.strip('.mp4') + '_' + model_name + '_QTD' + '_measure.txt', 'a') as f:
                    writer = csv.writer(f)
                    Data = str(datetime.datetime.now().strftime("%d/%m/%Y"))
                    Hora = str(datetime.datetime.now().strftime("%H:%M:%S"))
                    Segundo = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                    msegundo = int(cap.get(cv2.CAP_PROP_POS_MSEC) )
                    if antes == 0:
                       antes = int(time.time())
                    if antes >= agora - intervalo: # nesse caso deve fazer append na lista
                       intervalo_visitantes.append(visitantes)
                       agora = int(time.time())
                    else: # nesse caso deve sair com o MAX da lista e zerar a lista
                       maximo = max(intervalo_visitantes)
                       antes = int(time.time())
                       #print ( *intervalo_visitantes, sep=", ")
                       intervalo_visitantes=[]
                       intervalo_visitantes.append(visitantes)
                       #print("antes",antes)
                       csv_line = str(id) + ' , ' + Data + ' , ' + str(Segundo) + ' , ' + str(maximo) + ' , '  + Hora
                       id = id + 1
                       print( csv_line.split(',') )
                       writer.writerows([csv_line.split(',')])

                cv2.imshow('object counting',input_frame)

                output_movie.write(input_frame)
                #print("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

#                if (csv_line != "not_available"):
#                    with open(input_video.strip('.mp4') + '_' + model_name + '_QTD' + '_measure.txt', 'a') as f:
#                        writer = csv.writer(f)
#                        direction = csv_line.split(',')
#                        #print("csv_line:",csv_line )
#                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()


def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width,
                                 height):
    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    height = 0
    width = 0
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

    input_frame = cv2.imread(input_video)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(input_frame, axis=0)

    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # insert information text to video frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Visualization of the results of a detection.
    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1, input_frame,
                                                                                                 1,
                                                                                                 is_color_recognition_enabled,
                                                                                                 np.squeeze(boxes),
                                                                                                 np.squeeze(
                                                                                                     classes).astype(
                                                                                                     np.int32),
                                                                                                 np.squeeze(scores),
                                                                                                 category_index,
                                                                                                 use_normalized_coordinates=True,
                                                                                                 line_thickness=4)
    if (len(counting_mode) == 0):
        cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
    else:
        cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow('object counting', input_frame)
    cv2.waitKey(0)

    return counting_mode
