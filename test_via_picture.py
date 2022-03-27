import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model


def data_normalization(data, v2=True):  # 归一化，把范围变成[0,1](v2=False)或者[-1, 1](v2=True)区间内，好处：加快收敛、无量纲化
    if data.dtype != 'float32':
        data = data.astype('float32')
    data = data / 255.0
    if v2:
        data = data - 0.5
        data = data * 2.0
    return data


emotion_labels = {
    0: "高兴",
    1: "伤心",
    2: "中性",
}

emotion_model_path = './train/model_xce.h5'
emotion_classifier = load_model(emotion_model_path, compile=False)  # 导入模型

img = cv2.imread("./img/test.jpeg")  # 读取检测图像
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
face_detect = cv2.CascadeClassifier("Emotion_Classification/haarcascade_frontalface_default.xml")
faces_location = face_detect.detectMultiScale(gray_image, 1.3, 5)  # 人脸检测

input_shape = emotion_classifier.input_shape[1:3]  # 获取输入图片的大小，用于缩放图片
gray_faces = []
# 提取多个人脸到gray_faces中
for face_coordinates in faces_location:
    x1, y1, width, height = face_coordinates
    x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
    gray_face = gray_image[y1:y2, x1:x2]  # 裁剪图片
    try:
        gray_face = cv2.resize(gray_face, input_shape)  # 缩放图片
        gray_faces.append(gray_face)
    except:
        continue

gray_faces = np.asarray(gray_faces)
gray_faces = data_normalization(gray_faces, True)  # 归一化
gray_faces = np.expand_dims(gray_faces, -1)  # 拓展维度
emotion_prediction = emotion_classifier.predict(gray_faces)  # 调用模型预测

result = []
for emotion in emotion_prediction:
    emotion_label_index = np.argmax(emotion)  # 找出最大值的下标
    emotion_text = emotion_labels[emotion_label_index.item()]  # .item() 把ndarray(int)->python(int)
    result.append(emotion_text)

print("The shape of gray faces:", gray_faces.shape)
print("Output of mini xception:\n", emotion_prediction)
print("Result:", result)

fontpath = "./simsun.ttc"  # 新宋体字体文件
font_simsun = ImageFont.truetype(fontpath, 20)  # 加载字体, 字体大小
i = 0
for (x, y, w, h) in faces_location:
    # 图片，左上角坐标，右下角坐标，颜色BGR，矩形框粗细
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 在人脸上画矩形框
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色BGR，字体粗细；不支持中文
    # cv2.putText(img, result[i], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # 在矩形框上放置文字
    # cv2 putText不支持中文解决方案
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换PIL格式
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), result[i], font=font_simsun, fill=(0, 0, 0))  # (0,0,0)黑色, (255,255,255)白色
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)  # 转换回去
    i = i + 1

cv2.namedWindow('Emotion Detection of Humans(press q to quit)', cv2.WINDOW_NORMAL)  # 设置窗口大小
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
cv2.imshow('Emotion Detection of Humans(press q to quit)', img)  # 显示照片
cv2.waitKey(0)  # 按下任意按键退出
cv2.destroyWindow('Emotion Detection of Humans(press q to quit)')  # 关闭窗口
cv2.imwrite("Emotion_Classification/img/out.png", img)  # 保存图片
