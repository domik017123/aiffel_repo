# ResNet과 PlainNet 비교하기

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이동익
- 리뷰어 : 이효준


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
> 설계한 모델을 직접 Training과 시각화를 통해, 실제 학습과정을 확인할 수 있었습니다.
> 마지막 정확도 테스트에서 100점 만점을 통해 실제 학습한 경우에 더 좋은 성능이 보여짐이 확인 됐습니다.
 
- [x] 2.주석을 보고 작성자의 코드가 이해되었나요?  
```python
'''
(2) 라벨을 앵커박스로 인코딩 :
생성한 앵커박스와의 IoU가 0.5보다 높으면 물체, 0.4보다 낮으면 배경으로

'''
# IoU 계산
def convert_to_corners(boxes):
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

def compute_iou(boxes1, boxes2):
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

```  
>  네, 앵커박스의 IoU 기준값을 통해 어떻게 평가되는지 확인 됐습니다.

- [ ] 3.코드가 에러를 유발할 가능성이 있나요?
> 없습니다.

- [ ] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
def self_drive_assist(img_path, size_limit=300):
    image = Image.open(img_path).convert("RGB")
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    # 시각화
    visualize_detections(
        image,
        boxes,
        class_names,
        scores,
    )
    
    # 정지 조건   
    for box, class_name, score in zip(boxes, class_names, scores):
        if class_name == 'Pedestrian':
            print(class_name, 'STOP')
            return "Stop"
        if class_name == 'Car' or class_name == 'Van' or class_name == 'Truck':
            width = box[2]-box[0]
            height = box[3]-box[1]
            if width >= size_limit or height >= size_limit:
                print('Too close to the',class_name,'STOP')
                return "Stop"  
    print('GO')
    return 'Go'


self_drive_assist(img_path)
```
> 정지조건에 Stop이 return되고, 아닌경우 Go가 return되는 내용이 잘 구현되어 있습니다.
> 정지조건으로 사람이 한 명 이상 있는경우와 차량의 크기가 300px 넘는 경우를 잘 지켰습니다.

- [ ] 5.코드가 간결한가요?
```python
    # 정지 조건   
    for box, class_name, score in zip(boxes, class_names, scores):
        if class_name == 'Pedestrian':
            print(class_name, 'STOP')
            return "Stop"
        if class_name == 'Car' or class_name == 'Van' or class_name == 'Truck':
            width = box[2]-box[0]
            height = box[3]-box[1]
            if width >= size_limit or height >= size_limit:
                print('Too close to the',class_name,'STOP')
                return "Stop"  
    print('GO')
    return 'Go'
```
> Go, Stop 조건문 간결하게 잘 작성되었고 좋은 구현 아이디어 같습니다.
