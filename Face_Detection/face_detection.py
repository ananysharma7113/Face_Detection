import cv2
import logging as log
import datetime as dt
from time import sleep
import streamlit as st

casc_Path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_Cascade = cv2.CascadeClassifier(casc_Path)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

st.title('Welcome to the Face Detection App')

frame_placeholder = st.empty()

if st.button('Can I detect your Face?'):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            continue

        # capture frame by frame
        ret, frame = video_capture.read()

        if not ret:
            st.error('Failed to capture image from camera')
            break

        # grayscaling all the frames captured from the video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face from the grayscaled frames
        faces = face_Cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if anterior != len(faces):
            aneterior = len(faces)
            log.info('faces: '+str(len(faces)) + "at "+str(dt.datetime.now()))

        # Display the resulting frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Stopping the process on pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done release the capture
    video_capture.release()
    cv2.destroyAllWindows()
