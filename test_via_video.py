import numpy as np
import cv2
from tensorflow.keras.models import load_model


def preprocess_input(x, v2=True):
    if x.dtype != 'float32':
        x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


emotion_labels = {
    0: "happy",
    1: "sad",
    2: "neutral",
}

emotion_model_path = './train/model_xce.h5'
emotion_classifier = load_model(emotion_model_path, compile=False)  # 导入模型
input_shape = emotion_classifier.input_shape[1:3]  # 获取输入图片的大小，用于缩放图片
Face_detect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")  # 导入人脸检测模型

emojis = []
for emotion in emotion_labels.values():  # 取出emojis图片
    emojis.append(cv2.imread('./img/emojis/' + emotion + '.png', -1))
emojis = np.array(emojis).astype('float32')

video_path = "./img/test_video.mp4"
# video_path = 0
cap = cv2.VideoCapture(video_path)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
cv2.namedWindow('Emotion Detection of Humans(press q to quit)', cv2.WINDOW_NORMAL)  # 设置窗口大小
while True:
    ret, img = cap.read()  # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    if ret is False:
        break

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 转成灰度图
    faces_location = Face_detect.detectMultiScale(gray_image, 1.3, 5)  # 人脸检测
    if faces_location is not None:  # 检测到人脸
        for (x, y, w, h) in faces_location:  # 单独取出每一张人脸
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)    # 使用矩形，圈出人脸

            gray_face = gray_image[y:y+h, x:x+w]            # 裁剪图片
            gray_face = cv2.resize(gray_face, input_shape)  # 缩放图片
            gray_face = preprocess_input(gray_face, True)   # 标准化
            gray_face = np.expand_dims(gray_face, 0)        # 增加维度
            gray_face = np.expand_dims(gray_face, -1)       # 增加维度

            emotion_prediction = emotion_classifier.predict(gray_face)  # 调用模型，进行预测
            emotion_label_index = np.argmax(emotion_prediction)         # 找出最大值的下标
            emotion_text = emotion_labels[emotion_label_index.item()]   # 从字典中找出对应的表情
            cv2.putText(img, emotion_text, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 2)  # 在矩形框上放置文字

            for index in range(len(emotion_labels)):  # 在左上角使用条形图可视化
                cv2.putText(img, emotion_labels[index], (10, index * 20 + 20),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)
                cv2.rectangle(img, (130, index * 20 + 10),
                              (130 + int(emotion_prediction[0][index] * 100),
                               (index + 1) * 20 + 4), (255, 0, 0), -1)

            emojis_face = emojis[emotion_label_index.item()]
            for c in range(0, 3):                  # 左下角添加emojis表情，使用了边缘透明度，让GUI更好看
                img[200:320, 10:130, c] = emojis_face[:, :, c] * (emojis_face[:, :, 3] / 255.0) +\
                                          img[200:320, 10:130, c] * (1.0 - emojis_face[:, :, 3] / 255.0)

    cv2.imshow('Emotion Detection of Humans(press q to quit)', img)  # 展示结果
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

cap.release()
cv2.destroyWindow('Emotion Detection of Humans(press q to quit)')  # 关闭窗口
