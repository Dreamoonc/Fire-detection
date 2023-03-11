import torch
import cv2
import ssl
import beepy as beep

class FireDetection:
    def __init__(self,model_name):
        self.model=self.load_model(model_name)
        self.classes=self.model.names
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print("using device: ",self.device)
    
    def get_video_capture(self):
        return cv2.VideoCapture(0)
        #return cv2.VideoCapture('')

    def load_model(self , model_name):
        ssl._create_default_https_context = ssl.create_default_context
        model=torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        return model
    
    def score_frame(self,frame):
        self.model.to(self.device)
        frame=[frame]
        results=self.model(frame)
        labels, cord=results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]
        return labels,cord

    def class_to_label(self,x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results,frame):
        labels, cord = results
        n=len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row= cord[i]
            if row[4]>= 0:
                beep.beep(3)
                x1,y1,x2,y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape) , int(row[3]*y_shape)
                bgr=(0,0,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), bgr ,2)
                cv2.putText(frame , self.class_to_label(labels[i]), (x1, y1-10) ,cv2.FONT_HERSHEY_SIMPLEX, 0.9 ,(0,0,255) )
        return frame 
    
    def __call__(self):
        cap=self.get_video_capture()

        while(True):

            ret, frame = cap.read()
            results = self.score_frame(frame)
            if results is not None:
                labels, cord = results
                frame2 = self.plot_boxes(results, frame)
                n = len(labels)
                x_shape , y_shape = frame.shape[1] , frame.shape[0]
            else:
                frame2 = frame
            # Display the resulting frame
            results = self.score_frame(frame)
            frame2 = self.plot_boxes(results, frame)
            
            cv2.imshow('FIRE', frame2)
            
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

detector = FireDetection(model_name="best.pt")
detector()
