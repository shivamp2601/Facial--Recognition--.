from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture("sample.mp4")

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')         
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in bodies:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run()