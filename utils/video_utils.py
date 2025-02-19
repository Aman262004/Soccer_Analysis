import cv2

#Read the video frame by frame and return the list of frames
def read_video(video_path):
    #Video capture object with cv2 
    cap = cv2.VideoCapture(video_path)
    frames = []
    #Loops over the frames and adds it to the array 
    #Loops until ret = false which means the video has ended
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

#Takes the outputs video frames
def save_video(ouput_video_frames,output_video_path):
    #Define output format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Takes in video path which is just a string, takes the output video type, fps, and output frame width and height
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    #Loop over the frame and write it to the video out
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()