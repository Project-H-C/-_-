import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

def resize_images(input_folder, output_folder, target_size=(224, 224)):
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 이미지 파일의 리스트
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 이미지 조정 및 저장
    for file in tqdm(image_files):
        with Image.open(os.path.join(input_folder, file)) as img:
            # 이미지 크기 조정
            img = img.resize(target_size, Image.Resampling.LANCZOS)  # ANTIALIAS 대신 LANCZOS 사용
            img.save(os.path.join(output_folder, file))

def balance_data(source_folder_1, source_folder_2, num_samples, output_folder_1, output_folder_2):
    # 소스 폴더에서 이미지 파일 목록 가져오기
    images_1 = [f for f in os.listdir(source_folder_1) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images_2 = [f for f in os.listdir(source_folder_2) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 각 폴더에서 무작위로 num_samples 개의 이미지 선택
    images_1 = random.sample(images_1, min(num_samples, len(images_1)))
    images_2 = random.sample(images_2, min(num_samples, len(images_2)))

    # 이미지 파일을 대상 폴더로 복사
    for folder, image_list in zip([output_folder_1, output_folder_2], [images_1, images_2]):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for file in tqdm(image_list):
            shutil.copy(os.path.join(source_folder_1 if folder == output_folder_1 else source_folder_2, file), os.path.join(folder, file))

# 설정
num_samples = 3000
source_folder_216 = 'combined_images_216'
source_folder_311 = 'combined_images_311'
output_folder_216 = 'balanced_images_216'
output_folder_311 = 'balanced_images_311'

# 데이터 밸런싱
balance_data(source_folder_216, source_folder_311, num_samples, output_folder_216, output_folder_311)

# 이미지 크기 조정
resize_images(output_folder_216, 'resized_images_216')
resize_images(output_folder_311, 'resized_images_311')
