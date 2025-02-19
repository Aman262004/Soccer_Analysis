from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read video
    video_frames = read_video('Input_videos/soccer_clip.mp4')
    
    #Initalize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames)


    #Save video
    save_video(video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()


    '''
    Supervision format of bounding boxes
    Detections(xyxy=array([[     1371.2,       449.4,      1415.9,      523.87],
       [     539.91,      704.15,      600.27,      798.33],
       [     337.47,      737.32,       377.5,      845.33],
       [     238.93,      519.24,      274.41,      600.77],
       [       1230,      833.41,      1276.6,      928.48],
       [     676.99,      608.14,      713.76,       693.2],
       [     990.57,      659.65,      1044.8,      750.86],
       [     1308.2,      397.93,      1337.7,       469.8],
       [     357.64,      499.91,      386.33,      576.84],
       [     915.82,      366.29,      941.57,      430.22],
       [     1211.3,      356.37,      1241.5,      417.38],
       [     1111.2,      316.06,      1146.8,      363.64],
       [     1617.2,      654.28,      1667.7,       749.2],
       [     1027.4,      467.84,      1058.3,      537.63],
       [     374.61,      310.87,      399.25,      371.92],
       [     1272.3,      433.27,      1296.4,      505.96],
       [     783.96,      426.07,      812.79,       499.7],
       [     778.82,      375.14,      802.48,      434.93],
       [     963.27,      224.13,      982.82,      278.02],
       [     1852.3,      807.18,      1903.9,      915.91],
       [       1243,      716.87,      1294.9,      807.77],
       [     1184.5,      902.27,      1202.6,      921.12],
       [     319.67,      231.48,      340.33,      272.22],
       [     1241.7,      765.21,      1293.7,      839.78],
       [     1230.2,      769.38,      1278.8,         831],
       [     1243.4,      739.44,      1295.9,      838.91],
       [     991.35,      209.49,      1007.2,      238.38]], dtype=float32), mask=None, confidence=array([     0.9286,     0.92573,     0.92297,     0.92088,     0.91529,      0.9149,     0.90664,     0.90537,  
   0.90098,     0.89979,      0.8989,     0.89865,     0.89277,      0.8883,     0.88764,     0.88348,     0.87124,     0.85175,     0.85011,     0.84708,     0.70777,     0.68532,     0.66278,     0.44251,      
           0.34724,      0.2085,     0.11056], dtype=float32), class_id=array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 0, 3, 2, 2, 2, 3]), tracker_id=None, data={'class_name': array(['player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'referee', 'player', 'referee', 'player', 'ball', 'referee', 'player', 'player', 'player', 'referee'], dtype='<U7')})
    '''