import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np

# 입력 폴더와 출력 폴더 경로 설정
original_folder = 'dataset/test'
noisy_folder = 'robustness_test'

# SSIM 및 PSNR 계산 함수
def calculate_ssim_psnr(original_image, noisy_image):
    original = np.array(original_image)
    noisy = np.array(noisy_image)

    # 이미지 크기에 따라 win_size를 자동으로 설정
    min_side = min(original.shape[0], original.shape[1])
    win_size = min(7, min_side) if min_side >= 7 else min_side
    
    ssim_value = ssim(original, noisy, win_size=win_size, channel_axis=2)
    psnr_value = psnr(original, noisy)
    
    return ssim_value, psnr_value

# 전체 SSIM 및 PSNR 계산을 위한 변수 초기화
total_ssim = 0.0
total_psnr = 0.0
image_count = 0

# 0과 1 폴더의 모든 이미지에 대해 SSIM 및 PSNR 계산
for label_folder in ['0', '1']:
    original_label_folder = os.path.join(original_folder, label_folder)
    noisy_label_folder = os.path.join(noisy_folder, label_folder)
    
    for filename in os.listdir(original_label_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 원본 이미지와 노이즈가 추가된 이미지 파일 경로
            original_img_path = os.path.join(original_label_folder, filename)
            noisy_img_path = os.path.join(noisy_label_folder, filename)
            
            # 이미지 열기
            original_img = Image.open(original_img_path).convert('RGB')
            noisy_img = Image.open(noisy_img_path).convert('RGB')
            
            # SSIM 및 PSNR 계산
            ssim_value, psnr_value = calculate_ssim_psnr(original_img, noisy_img)
            
            # 개별 SSIM 및 PSNR 출력
            print(f"File: {label_folder}/{filename} - SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB")
            
            # 총합 계산
            total_ssim += ssim_value
            total_psnr += psnr_value
            image_count += 1

# 평균 SSIM 및 PSNR 계산
average_ssim = total_ssim / image_count if image_count > 0 else 0
average_psnr = total_psnr / image_count if image_count > 0 else 0

# 전체 평균 출력
print(f"\nAverage SSIM: {average_ssim:.4f}")
print(f"Average PSNR: {average_psnr:.2f} dB")
