from io import BytesIO
from time import time
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import base64
import moviepy.editor as moviepy
import time
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import os

import base64
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime



CLASS_NAME_FILE = "model/coco.names"
WEIGHT_FILE = "model/yolov4-tiny.weights"
CLF_FILE = "model/yolov4-tiny.cfg"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 1

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

################################################
# FUNCTION

@st.cache(allow_output_mutation=True)
def load_model(weights=WEIGHT_FILE,clf=CLF_FILE):
    print("Load YOLOv4 model: Done")
    net = cv2.dnn.readNet(weights,clf)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    return net , model

@st.cache(allow_output_mutation=True)
def loadModel(weights=WEIGHT_FILE):
    print("Load YOLOv5 model: Done")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True) 
    return model

def get_classes(file_name=CLASS_NAME_FILE):
    classes = []
    with open(file_name, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def get_output_layer(net):
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return output_layers



def get_image_shape(img):
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape
    return height, width, channels


def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs



def detection_inferrence(outs,threshold):
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold:

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return class_ids, boxes, confidences



def non_maximum_suppresion(boxes, confidences):
    return cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)




def display(boxes,indexes,img):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = [int(i) for i in colors[class_ids[i]]]
            confi = confidences[i]

 
            cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=THICKNESS)
            
            text = f"{label}: {confi:.2f}"
            (text_width, text_height) = cv2.getTextSize(text, FONT,
             fontScale=FONT_SCALE, thickness=THICKNESS)[0]

            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))

            overlay = img.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            img=cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            
            cv2.putText(img, text, (x, y - 5), FONT,
            fontScale=FONT_SCALE, color=(0, 0, 0), thickness=THICKNESS)

    return img

def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href



class Video(VideoProcessorBase):

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            Classes, scores, boxes = model.detect(
                image, 0.4, 0.4)
            for (classid, score, box) in zip(Classes, scores, boxes):

                color = COLORS[int(classid) % len(COLORS)]

                label = "%s : %f" % (classes[classid], score)

                cv2.rectangle(image, box, color, 1)
                cv2.putText(image, label, (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
  


########################################################################

st.title("YOLO MODEL APPLICATION")

bl3,header,bl4 = st.columns(3)
header.header("Final project")

st.subheader("Information")
st.write("19127082 - Nguyễn Tất Trường \n\n\
19127562 - Chung Thế Thọ")

read_me = st.markdown("""
    This project was built using Streamlit, OpenCV, Pytorch
    to demonstrate YOLO Object detection for images,videos(pre-recorded)
    and camera(for YOLOv4 only).
    
    In YOLOv4: this YOLO object Detection project can detect 80 objects(i.e classes)
    in either a video or image or camera. 
    
    In YOLOv5: this YOLO model we use vehicles dataset for training. The dataset contents object of 6 classes.
    The model can detect in  either a video or image."""
    )

# class of detection
classes = []

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

ver = st.sidebar.selectbox("YOLO version",["YOLOv4","YOLOv5"])
if ver == "YOLOv4":
    rad = st.sidebar.radio("Detection options",["Image","Video","Camera"])

    # Load darknet files YOLO, model
    net, model = load_model(WEIGHT_FILE,CLF_FILE)


    classes = get_classes(CLASS_NAME_FILE)
    output_layers = get_output_layer(net)


    # generate color list
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
      
            
    # Image detection
    if rad =="Image":
        st.subheader("Image option")
        uploaded_file = st.file_uploader("Choose a file", key=0)
    
        if uploaded_file is not None:

            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            
            st.write("Original picture")
            st.image(img,width=700)
            height, width, channels = get_image_shape(img)
            # Detecting objects
            outs = detect_objects(img)
            
            st.subheader("Setting the threshold for our detection")
            threshold = st.slider("Threshold", min_value=0.00, max_value=1.0, step=0.05, value=0.5)
            if st.button("Detect"):
                class_ids, boxes, confidences = detection_inferrence(outs,threshold)
                # remove noise
                indexes = non_maximum_suppresion(boxes, confidences)
                        
                st.write(f"Object detection with confidence: {threshold:.2f}")
                img = display(boxes,indexes,img)
                
                ## display result
                st.image(img,width=700)
                result = Image.fromarray(img)
                
                st.markdown(get_image_download_link(result,"result.jpg",'Download '+"result.jpg"), unsafe_allow_html=True)
        
    #Video detection
    elif rad =="Video":
        st.subheader("VIDEO option")
        with st.form("my-form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Choose a file", key=1)
            col1, col2, col3 = st.columns([3,2,1])

            with col3:
                submitted = st.form_submit_button("Upload")

        if submitted and uploaded_file is not None:
            vid = uploaded_file.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_file.read()) # save video to disk
            
            st_video = open(vid,'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            cap = cv2.VideoCapture(vid)
            _, image = cap.read()
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            height, width, channels = get_image_shape(image)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (width, height))
            count = 0
            p = st.empty()
            p2 = st.empty()
            begin = time.perf_counter()
            while True:
                _, image = cap.read()
            
                if  _!= False:
                    try:
                        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
                        height, width, channels = get_image_shape(image)
                        start = time.perf_counter()
                        outs = detect_objects(image)
                        time_took = time.perf_counter() - start
                        count +=1
                        p.write(f"Time took: {count} {time_took}")
                        
                        class_ids.clear()
                        boxes.clear()
                        confidences.clear()
                        class_ids, boxes, confidences = detection_inferrence(outs,0.5)
                        indexes = non_maximum_suppresion(boxes, confidences)
                        image = display(boxes,indexes,image)
                        image = cv2.cvtColor(image , cv2.COLOR_RGBA2BGR)
                        out.write(image)
                    except KeyboardInterrupt:
                        break
                else:
                    break
            
            p.write(f"Rendering time: {(time.perf_counter() - begin):.2f}")
            cap.release()
            out.release()
            try:
                clip = moviepy.VideoFileClip('detected_video.mp4')
                clip.write_videofile("myvideo.mp4")
                st_video = open('myvideo.mp4','rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("Detected Video") 
            except OSError:
                print("error with moviepy")

            # clip = moviepy.VideoFileClip('detected_video.mp4')
            # clip.write_videofile("myvideo.mp4")
            # st_video = open('myvideo.mp4','rb')
            # video_bytes = st_video.read()
            # st.video(video_bytes)
            # st.write("Detected Video") 
        
    # Camera detection
    elif rad =="Camera":
        st.subheader("Camera option")

        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            
            rtc_configuration={"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}]},
            
            media_stream_constraints={
            "video": True,
            "audio": False,},
            
            video_processor_factory=Video,
            async_processing=True,
        )

    else:
        st.write("No file was choosen!")
else:
    rad = st.sidebar.radio("Detection options",["Image","Video"])
    model = loadModel('model/best.pt') 
    if rad == "Image":    
        image_file = st.file_uploader("Choose a file", key=2)
    
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, width=700, caption='Uploaded Image')

            st.subheader("Setting the threshold for our detection")
            threshold = st.slider("Threshold", min_value=0.00, max_value=1.0, step=0.05, value=0.5)

            if st.button("Detect"):                       
                st.write(f"Object detection with confidence: {threshold:.2f}")
                #call Model prediction--
                model = loadModel('model/best.pt')
                model.conf = threshold  
                pred = model(img)
                pred.render()  # render bbox in image
                
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                #im_base64.save(outputpath)

                #--Display predicton
                    
                #img_ = Image.open(im_base64)
                st.image(im_base64, width=700, caption='Model Prediction(s)')
                    
                st.markdown(get_image_download_link(im_base64,"result.jpg",'Download '+"result.jpg"), unsafe_allow_html=True)
    
    elif rad == "Video":
        st.subheader("VIDEO option")
        
        with st.form("my-form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Choose a file", key=3)
            col1, col2, col3 = st.columns([3,2,1])

            with col3:
                submitted = st.form_submit_button("UPLOAD!")

        if submitted and uploaded_file is not None:
            model.conf = 0.5
            vid = uploaded_file.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_file.read()) # save video to disk
                
            st_video = open(vid,'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Uploaded Video")

            cap = cv2.VideoCapture(vid)
            _, image = cap.read()
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            height, width, channels = get_image_shape(image)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (width, height))
            count = 0

            p = st.empty()
            p2 = st.empty()
            begin = time.perf_counter()
            while True:
                _, image = cap.read()
                    

                if  _!= False:
                    try:
                        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
                        height, width, channels = get_image_shape(image)

                        start = time.perf_counter()

                        pred = model(image)
                        pred.render()  # render bbox in image

                        time_took = time.perf_counter() - start
                        count +=1
                        p.write(f"Time took: {count} {time_took}")

                        image = cv2.cvtColor(np.array(pred.ims)[0] , cv2.COLOR_RGBA2BGR)
                        out.write(image)
                    except KeyboardInterrupt:
                        break
                else:
                    break
                
            p.write(f"Rendering time: {(time.perf_counter() - begin):.2f}")


            cap.release()
            out.release()

            try:
                clip = moviepy.VideoFileClip('detected_video.mp4')
                clip.write_videofile("myvideo.mp4")
                st_video = open('myvideo.mp4','rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("Detected Video") 
            except OSError:
                ''




