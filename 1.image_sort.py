import os
import shutil

def combine_images_from_folders(base_path, result_folders, target_folders, output_folder):
    # 출력 폴더가 없다면 생성
    os.makedirs(output_folder, exist_ok=True)
    for result in result_folders:
        for target_folder in target_folders:
            current_path = os.path.join(base_path, result, target_folder)
            if os.path.exists(current_path):
                for image_name in os.listdir(current_path):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(current_path, image_name)
                        # 이미지 파일을 출력 폴더로 복사
                        shutil.copy(image_path, os.path.join(output_folder, f"{result}_{target_folder}_{image_name}"))

# 설정
data_folders = ['표지판코드분류crop데이터1', '표지판코드분류crop데이터2']
result_folders = ['result_1', 'result_2', 'result_3', 'result_6', 'result_7', 'result_8-1', 'result_8-2', 'result_8-3', 'result_8-4', 'result_8-5', 'result_8-6', 'result_8-7', 'result_8-8', 'result_8-9', 'result_8-10', 'result_8-11', 'result_8-12', 'result_8-13', 'result_9', 'result_11', 'result_12', 'result_14', 'result_15', 'result_17', 'result_18', 'result_19', 'result_20', 'result_21', 'result_22']
target_folders = ['216', '311']
output_folder_216 = 'combined_images_216'  # Directory for combined images 216
output_folder_311 = 'combined_images_311'  # Directory for combined images 311

# 각 데이터 폴더에 대해 이미지 복사 작업 수행
for data_folder in data_folders:
    # base_path = os.path.join(data_folder, data_folder)  # Robustness\표지판코드분류crop1\표지판코드분류crop1\result...으로 구성하였을 경우
    base_path = data_folder
    combine_images_from_folders(base_path, result_folders, ['216'], output_folder_216)
    combine_images_from_folders(base_path, result_folders, ['311'], output_folder_311)
    
