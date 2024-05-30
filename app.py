import onnxruntime as ort
import numpy as np
import cv2
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

# Model saqlangan papkaga yo'l
model_path = 'models/best.onnx'

# ONNX Runtime session paremetrlarini kiritamiz
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = True
session_options.enable_cpu_mem_arena = True
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Provayderlar 
EP_list = ['CPUExecutionProvider','CUDAExecutionProvider']

# modelni yuklab olish
try:
    ort_session = ort.InferenceSession(model_path, sess_options=session_options, providers=EP_list)
    print(f"Model muvaffaqiyatli yuklandi {model_path}")
except Exception as e:
    print(f"Modelni yuklashda xatolik: {e}")
    exit()

# rasmni o'lchamlarini to'g'irlab olamiz
def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = img.transpose(2, 0, 1)  
    img = img[np.newaxis, :, :, :].astype(np.float32)  
    img /= 255.0  
    return img

# Chiqish ma'lumotlari bilan ishlash(bbox, scores)
def postprocess(outputs, img_shape):
    boxes, scores = [], []
    detections = outputs[0]
    for detection in detections[0].T:
        confidence = detection[4]
        if confidence > 0.5:
            x_center, y_center, width, height = detection[:4]
            x1 = int((x_center - width / 2) * img_shape[1] / 640)
            y1 = int((y_center - height / 2) * img_shape[0] / 640)
            x2 = int((x_center + width / 2) * img_shape[1] / 640)
            y2 = int((y_center + height / 2) * img_shape[0] / 640)
            x1 = max(0, min(x1, img_shape[1]))
            y1 = max(0, min(y1, img_shape[0]))
            x2 = max(0, min(x2, img_shape[1]))
            y2 = max(0, min(y2, img_shape[0]))
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)

    # Non-Maximum Suppression (NMS)
    index = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)
    print(f"NMS indices: {index}")
    if len(index) > 0:
        nms_boxes = [boxes[i] for i in index.flatten()]
        nms_scores = [scores[i] for i in index.flatten()]
    else:
        nms_boxes = []
        nms_scores = []

    return nms_boxes, nms_scores

# bboxlarni chizamiz
def draw_boxes(image, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Flask kamera uchun
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        input_image = preprocess(frame)
        ort_inputs = {ort_session.get_inputs()[0].name: input_image}
        ort_outs = ort_session.run(None, ort_inputs)
        boxes, scores = postprocess(ort_outs, frame.shape)
        draw_boxes(frame, boxes, scores)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
