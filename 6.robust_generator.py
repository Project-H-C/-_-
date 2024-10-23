import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# 모델 로드
model = load_model('custom_cnn_model.keras')
model.summary()

# 이미지 전처리 설정
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # [0, 1] 범위로 스케일링
    return img

# FGSM 공격 함수
def fgsm_attack(image, epsilon, gradient):
    sign_data_grad = tf.sign(gradient)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    return perturbed_image

# 입력 폴더와 출력 폴더 경로 설정
input_folder = 'dataset/test'
output_folder = 'robustness_test'
os.makedirs(output_folder, exist_ok=True)

# 0과 1 폴더에 대해 FGSM 공격 적용
for label_folder in ['0', '1']:
    input_label_folder = os.path.join(input_folder, label_folder)
    output_label_folder = os.path.join(output_folder, label_folder)
    os.makedirs(output_label_folder, exist_ok=True)
    
    for filename in os.listdir(input_label_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_label_folder, filename)
            img = preprocess_image(img_path)
            
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                prediction = model(img_tensor)
                loss = tf.keras.losses.binary_crossentropy(tf.ones_like(prediction), prediction) # 이진 분류 시 
                # loss = tf.keras.losses.categorical_crossentropy(prediction, prediction) # 다중 분류 시
            gradient = tape.gradient(loss, img_tensor)

            epsilon = 0.01
            perturbed_img = fgsm_attack(img_tensor, epsilon, gradient)
            perturbed_img = np.clip(perturbed_img[0] * 255, 0, 255).astype(np.uint8)  # [0, 1] -> [0, 255]

            perturbed_img_pil = Image.fromarray(perturbed_img)
            output_path = os.path.join(output_label_folder, filename)
            perturbed_img_pil.save(output_path)

print("0과 1 폴더의 모든 이미지에 FGSM 공격이 적용되어 저장되었습니다.")
