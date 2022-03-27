import cv2
from deepface import DeepFace
import time
from flask import Flask, render_template, Response
app = Flask(__name__)

def images():
    frame = cv2.imread("C:/Users/Abhineet/Documents/DS 360 DigiTMG/Drowsiness Detection Project_Feb-22/disgust.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = DeepFace.analyze(frame, actions=['emotion'])
    cv2.putText(frame, predictions['dominant_emotion'], (20, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2)
    x, y, w, h = int(predictions['region']['x']), int(predictions['region']['y']), int(predictions['region']['w']), int(predictions['region']['h'])
    cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 1)
    #frame = cv2.resize(frame, (700, 500))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/output')
def output():
    return Response(images(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(host = '127.0.0.1', port = 5000, debug=True, use_reloader = True)