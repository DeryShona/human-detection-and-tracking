
import cv2
import imutils
import numpy as np
import argparse

#Detect method frame by frame
#Counts the number of people in the video
def detect(frame):
  bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
  person = 1
  for x,y,w,h in bounding_box_cordinates:
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    person += 1
  cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
  cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
  cv2.imshow('output', frame)
  return frame

#Reads from the web camera
#Detects method fram by frame
def detectFromVideoFeed(path, writer):
  video = cv2.VideoCapture(path)

  print('Detecting people...')
  while True:
    check, frame = video.read()
    frame = detect(frame)
    if writer is not None:
      writer.write(frame)
    else:
      cv2.imshow('Video Feed', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
      break
  video.release()
  cv2.destroyAllWindows()

#Creating a human detector 

def humanDetector(args):
  writer = None
  if args['output'] is not None:
    writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    
  print('[INFO] Opening Web Cam.')
  detectFromVideoFeed(0, writer)

def argsParser():
  arg_parse = argparse.ArgumentParser()
  arg_parse.add_argument(
    "-o",
    "--output",
    type=str,
    help="path to optional output video file",
    nargs="?",
    default="video.mp4",
    const="'video.mp4"
  )
  args = vars(arg_parse.parse_args())

  return args
  
if __name__ == "__main__":

#Creating a model which will detect human
  HOGCV = cv2.HOGDescriptor()
  HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

  args = argsParser()
  humanDetector(args)