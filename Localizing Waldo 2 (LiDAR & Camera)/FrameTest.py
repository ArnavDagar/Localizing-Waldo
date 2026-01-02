import cv2

cap = cv2.VideoCapture('Data/test_w21.h264')
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(frames,fps)
print(frames/fps)

"""
frame_no = 0
while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_FPS)))
    else:
        break
    frame_no += 1
"""
cap.release()


import imageio
filename="Data/test_w21.h264"
vid = imageio.get_reader(filename,  'ffmpeg')
# number of frames in video
num_frames=vid._meta['nframes']
print(num_frames)
