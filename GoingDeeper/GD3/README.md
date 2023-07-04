# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.06.30]

- 코더 : 이동익
- 리뷰어 : 부석경

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
|평가문항|상세기준|완료여부|
|-------|---------|--------|
|1. CAM을 얻기 위한 기본모델의 구성과 학습이 정상 진행되었는가?|ResNet50 + GAP + DenseLayer 결합된 CAM 모델의 학습과정이 안정적으로 수렴하였다.| ![image](https://github.com/domik017123/aiffel_repo/assets/71332005/ffedf031-d85a-4422-8e96-a2fb185ddb14) |
|2. 분류근거를 설명 가능한 Class activation map을 얻을 수 있는가?|CAM 방식과 Grad-CAM 방식의 class activation map이 정상적으로 얻어지며, 시각화하였을 때 해당 object의 주요 특징 위치를 잘 반영한다.| * CAM Image   ![image](https://github.com/domik017123/aiffel_repo/assets/71332005/2947228d-1eee-4230-97fb-cfa3e32601c3) * Grad-CAM   ![image](https://github.com/domik017123/aiffel_repo/assets/71332005/4ba99aba-6b2c-4d5d-bf9c-392a6477a7dd)|
|3. 인식결과의 시각화 및 성능 분석을 적절히 수행하였는가?|CAM과 Grad-CAM 각각에 대해 원본이미지합성, 바운딩박스, IoU 계산 과정을 통해 CAM과 Grad-CAM의 object localization 성능이 비교분석되었다.|![image](https://github.com/domik017123/aiffel_repo/assets/71332005/4cef1d2b-9e24-4b00-95c4-b539439f28e4)|  

* 3번의 Grad-CMA의 비교가 빠져있습니다.

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**

(3)CAM 구현

```python
cam_model = tf.keras.models.load_model('./aiffel/class_activation_map/cam_model')

def generate_cam(model, item):
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    
    img_tensor, class_idx = normalize_and_resize_img(item) #(224, 224, 3),()
    
    # 학습한 모델에서 원하는 Layer의 output을 얻기 위해서 모델의 input과 output을 새롭게 정의해줍니다.
    cam_model = tf.keras.Model([model.inputs], #(1, 224, 224, 3)
                               [model.layers[-3].output, model.output])#(1, 7, 7, 2048),(1, 120)
    conv_outputs, predictions = cam_model(tf.expand_dims(img_tensor, 0)) #img_tensor에 배치차원 추가
    conv_outputs = conv_outputs[0, :, :, :] #(7, 7, 2048)
    
    # 모델의 weight activation은 마지막 layer에 있습니다.
    class_weights = model.layers[-1].get_weights()[0] #(2048,120)
    cam_image = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2]) #(7,7)
    
    for i, w in enumerate(class_weights[:, class_idx]):#(2048,120)에서 class_idx에 해당하는 weights선택
        # conv_outputs의 i번째 채널과 i번째 weight를 곱해서 누적하면 활성화된 정도가 나타날 겁니다.
        cam_image += w * conv_outputs[:, :, i]# (i번째 채널의 weights)*(i번째 채널의 7x7 conv_outputs), 채널 수 만큼(2048번) 반복

    cam_image /= np.max(cam_image) # activation score를 normalize합니다.
    cam_image = cam_image.numpy() #(7,7)
    plt.imshow(cam_image)
    plt.show()
    cam_image = cv2.resize(cam_image, (width, height)) # 원래 이미지의 크기로 resize합니다.
    return cam_image
```

* 각 변수의 크기를 적어주고 주석을 너무 잘 달아주었습니다. 위 코드 이외에도 설명이 아주 잘 되있어 이해가 매우 쉬었습니다.

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
* 발견하지 못했습니다.

### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
```python
'''
(1)바운딩 박스 구하기
'''
def get_bbox(cam_image, score_thresh=0.2):
    # th 이하 0으로 날리기
    low_indicies = cam_image <= score_thresh
    cam_image[low_indicies] = 0
    cam_image = (cam_image*255).astype(np.uint8)
    
    # cv2.findContours로 bbox 얻기
    '''
    cv2.findContours()        : 값이 동일한 등고선 그리기
    cv2.RETR_TREE             : 등고선을 계층적으로 구성하는 옵션
        > contours[0]         : 가장 바깥쪽 등고선
    cv2.CHAIN_APPROX_SIMPLE   : contours line을 그릴 수 있는 최소 point만 저장 
    cv2.minAreaRect(cnt)      : 회전 고려, 가장 작은 bbox 얻기(x,y,w,h,각도)
    cv2.boxPoints             : bbox 꼭지점 얻기
    '''
    contours,_ = cv2.findContours(cam_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rotated_rect = cv2.minAreaRect(cnt)
    rect = cv2.boxPoints(rotated_rect)
    rect = np.int0(rect) #정수로 변환
    return rect
```
* 위 주석과 같이 모든 코드 라인을 모두 이해하시려고 공부하시고 작성해 주셔서 제대로 이해하고 작성한 코드입니다.

### **[⭕] 코드가 간결한가요?**
* 네, 불필요한 코드는 없다고 생각합니다.

----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
