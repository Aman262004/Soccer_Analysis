Soccer Analysis:

This application analyzes video clips from soccer matches to accurately distinguish and identify objects such as players, the ball, and referees. It utilizes the YOLOv8 object detection model, which will be enhanced by training with a dataset from Roboflow to improve its accuracy and performance.

Raw Input:
![](images/Raw_input_clip.jpg)

To begin the analysis, we first need to install the Ultralytics library and load the YOLOv8 model, specifically yolov8s. There are different versions of the model, such as yolov8m and yolov8x, where the letters 's', 'm', and 'x' represent the model's capabilities 's' for small, 'm' for medium, and 'x' for a more powerful model.

The choice of model depends on your PC’s specifications. Once the appropriate YOLO model is selected, we can run the object detection process. The model will detect objects in the video, drawing bounding boxes around them and assigning confidence scores.

It is important to note that all players and referees are classified as 'persons' by default. This is because the pre-trained YOLO model does not differentiate between different types of people, whether they are soccer players, basketball players, or referees—they are all identified simply as 'persons'.

Once you we run the model on the input video we will get the following output:
![](images/output_1.jpg)

We can see how the model actually creates the bounding box with the following output:

''' EXAMPLE OF BINDING BOX For yolov8s model (Overall detection is not that good as the object detection is limited by the amount of registerd objects)
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
'''

To improve the model and enable it to distinguish referees from players, we will use a sample dataset provided by Roboflow. The setup process is detailed in the soccer_training_yolo_v5.ipynb file.

We start by installing the necessary libraries, then downloading the dataset from Roboflow. Finally, we train the object detection model. For the most accurate results, we train the model for 100 epochs.

Before discussing the output, let's understand how the data is processed. Each frame of the video is fed into the model, and for each frame, there is a labeled identification that corresponds to every player. This approach falls under supervised learning, where the dataset consists of labeled examples that guide the model’s learning process.

Below is an example of frame and its corresponding object idetnfication labels

Frame:
![](images/labeled_data_1.jpg)

Labeled Data:
![](images/labeled_data_2.jpg)

Now we understand how the training works we can finally take a look at the final output once the model has undergone training

Output once the model has undergone training:


