import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모델 로드
model = load_model('custom_cnn_model.keras')

# 테스트 데이터 로드
#test_dir = 'dataset/test'
test_dir = 'robustness_test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,  # 배치 크기를 1로 설정하여 각 이미지를 개별적으로 평가
    class_mode='binary', # 이진 분류일 경우
    # class_mode='categorical', # 다중 분류일 경우
    shuffle=False)

# 예측 수행 및 정확도 평가
predictions = model.predict(test_generator, steps=test_generator.samples)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1) # 이진 분류일 경우
#predicted_classes = np.argmax(predictions, axis=1) # 다중 분류일 경우
true_classes = test_generator.classes
accuracy = np.mean(predicted_classes == true_classes)
print(f'Test Accuracy: {accuracy:.4f}')

