# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 이동익
- 리뷰어 : 장승우

----------------------------------------------

**PRT(PeerReviewTemplate)**

** [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
네~ 그래프와 모델 훈련 정상적으로 진행되었어요~

```pyhon
![image](https://github.com/domik017123/aiffel_repo/assets/131636630/901fef04-8e36-4efc-9c1f-147466618735)
```
** [O] 주석을 보고 작성자의 코드가 이해되었나요?
네~ 주석을 상세하게 달아주셔서 이해가 쉬웠어요~
```python
def apply_preprocessing(ds, is_test=False, batch_size=BATCH_SIZE, 
                               with_aug=False, with_cutmix=False, with_mixup=False):
    
    ds = ds.map(normalize_and_resize_img, num_parallel_calls=AUTOTUNE)#all ds resize,rescale
    
    if not is_test and with_aug:#train_aug,cutmix,mixup에서 augmentation
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        
    ds = ds.batch(batch_size)#all ds into batch
    
    if not is_test and with_cutmix:#train_cutmix / cutmix 과정에서 one-hot 진행
        ds = ds.map(cutmix, num_parallel_calls=AUTOTUNE)
    elif not is_test and with_mixup:#train_mixup / mixup 과정에서 one-hot 진행
        ds = ds.map(mixup, num_parallel_calls=AUTOTUNE) 
    else:#train_no_aug, val, test / one-hot 진행
        ds = ds.map(onehot, num_parallel_calls=AUTOTUNE)
        
    if not is_test:#train ds만 repeat/shuffle
        ds = ds.repeat() #훈련 데이터셋 계속 입력 / 이후 steps_per_epoch을 통해 한 epoch을 종료
        ds = ds.shuffle(SHUFFLE_SIZE) #파일 순서가 학습에 영향을 주지 않도록 셔플
        
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)#all ds prefetch / 입력,훈련 동시 진행
    
    return ds
```
** [X] 코드가 에러를 유발할 가능성이 있나요?
변수처리를 해주셔서 유발할 가능성이 낮을 것 같아요~
```python
IMG_SIZE = 128
BATCH_SIZE = 16
NUM_CLASS = 120

#Resize and normalize image
def normalize_and_resize_img(image, label):
    # Normalizes images: `uint8` -> `float32`
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
```

** [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
네~ 질문에 대한 대답과 코드도 직접 변경하시면서 진행하셨어요~
```python
# image_a,b의 면적 비율에 맞게 라벨 변환
def mix_2_labels(image_a, image_b, label_a, label_b, x_min, y_min, x_max, y_max, num_classes=NUM_CLASS):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]
```

** [O] 코드가 간결한가요?
네~ 특히 mixup 부분처리를 간결하게 elif로 처리해주셨어요~
```python
    if not is_test and with_cutmix:#train_cutmix / cutmix 과정에서 one-hot 진행
        ds = ds.map(cutmix, num_parallel_calls=AUTOTUNE)
    elif not is_test and with_mixup:#train_mixup / mixup 과정에서 one-hot 진행
        ds = ds.map(mixup, num_parallel_calls=AUTOTUNE) 
    else:#train_no_aug, val, test / one-hot 진행
        ds = ds.map(onehot, num_parallel_calls=AUTOTUNE)
        
```


----------------------------------------------

참고 링크 및 코드 개선
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```python
mixup 을 elif로 넣는 부분과 같이 깔끔하게 작성하는 방법을 배웠어요~
논문의 이미지도 참고해주시면서 설명을 같이 적어주셔서 좋았어요~
또한 과제 진행하면서 실패했던 부분도 같이 적어주셔서 도움이 되었어요~
```
