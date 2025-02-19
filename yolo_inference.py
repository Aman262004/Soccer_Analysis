from ultralytics import YOLO

#Importing object detection model
#model = YOLO('yolov8s')
model = YOLO('models/best.pt')

#Running the model and saving the result
results = model.predict('input_videos/soccer_clip.mp4', save=True)

#Goes through the frames and puts bounding box and confidence of object
print(results[0])
print('==============')
for box in results[0].boxes:
    print(box)

''' EXAMPLE OF BINDING BOX For yolov8s model (Overall accuracy is not that good)
cls: tensor([0.]) -----  0 means person
conf: tensor([0.7660]) ---- Confidence of the object being an person
data: tensor([[5.3302e+02, 6.8714e+02, 5.7907e+02, 7.8698e+02, 7.6599e-01, 0.0000e+00]])
id: None
is_track: False
orig_shape: (1080, 1920)
shape: torch.Size([1, 6])
xywh: tensor([[556.0414, 737.0588,  46.0479,  99.8365]]) ---- cords of bounding box
xywhn: tensor([[0.2896, 0.6825, 0.0240, 0.0924]])
xyxy: tensor([[533.0175, 687.1406, 579.0654, 786.9771]])
xyxyn: tensor([[0.2776, 0.6362, 0.3016, 0.7287]])
ultralytics.engine.results.Boxes object with attributes:

To make the model better we use a training trail on YOLO and fine tune it using a data set from Roboflow
'''