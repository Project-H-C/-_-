import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_datasets(input_folders, labels, base_dir='dataset'):
    # 폴더 구조 생성
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        for label in labels:
            path = os.path.join(base_dir, subset, str(label))
            os.makedirs(path, exist_ok=True)

    # 데이터 로드 및 분할
    for folder, label in zip(input_folders, labels):
        image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        file_paths = [os.path.join(folder, f) for f in image_files]

        # 데이터 분할
        train_files, test_files = train_test_split(file_paths, test_size=0.3, random_state=42)
        valid_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

        # 데이터 저장
        for files, subset in zip([train_files, valid_files, test_files], subsets):
            for file in files:
                shutil.copy(file, os.path.join(base_dir, subset, str(label), os.path.basename(file)))

# 이미지 폴더 설정
input_folders = ['resized_images_216', 'resized_images_311']
labels = [0, 1]  # 0: resized_images_216, 1: resized_images_311

# 데이터셋 준비
prepare_datasets(input_folders, labels)
