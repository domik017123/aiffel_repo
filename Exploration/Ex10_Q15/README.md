## Code Peer Review Template
---
* 코더 : 이동익
* 리뷰어 : 정연준


## PRT(PeerReviewTemplate)
---
- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

![image](https://github.com/domik017123/aiffel_repo/assets/131635437/a597eb6d-a3d7-40c9-a99a-3cb60bd8eea6)
![image](https://github.com/domik017123/aiffel_repo/assets/131635437/0418d732-219f-418a-b705-f6f4e758e84b)

- [x] 주석을 보고 작성자의 코드가 이해되었나요?

각 함수별로 어떤 코드들이 어떤 기능을 수행하고 있는지 상단에 주석으로 처리하여 이해할 수 있었다.

```python
#blurring function
def apply_segmentation_mask(_img_path, _target_class):
    img = cv2.imread(_img_path)
    target_class = _target_class
    
    # Adjust image segmentation
    segvalues, output = model.segmentAsPascalvoc(_img_path)

    # Get class
    for class_id in segvalues['class_ids']:
        print(LABEL_NAMES[class_id],class_id)

    # Get segmentation map of target class (T/F)
    print('colormap: ',colormap[target_class])
    seg_color = (colormap[target_class][2], colormap[target_class][1], colormap[target_class][0])
    seg_map = np.all(output == seg_color, axis=-1)

    # Blurring image
    img_org_blur = cv2.blur(img, (30, 30))

    # Mask: True = 255, False = 0
    img_mask = seg_map.astype(np.uint8) * 255
    img_mask_bgr = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

    # Background mask: background: 255, target class: 0
    img_bg_mask = cv2.bitwise_not(img_mask_bgr)

    # Blurring background: blurred img + background mask
    img_bg_blur = cv2.bitwise_and(img_org_blur, img_bg_mask)

    # Image + blurred background
    img_concat = np.where(img_mask_bgr == 255, img, img_bg_blur)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1,3,2)
    plt.imshow(output)
    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(img_bg_mask, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return img_concat
```

- [x] 코드가 에러를 유발할 가능성이 있나요?

- 함수로 묶어구성하여 에러가 발생할 여지를 어느정도 줄이고, 전체적으로 간결하게 구성하였다(상세 코드는 아래 기재)

- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

- 각 코드가 어떤 의미인지 인지하고, 결과물을 출력하여 함수가 구동될때 각 출력의 결과물의 의미를 출력하도록 코드를 구성했다.

```python
    # Get class
    for class_id in segvalues['class_ids']:
        print(LABEL_NAMES[class_id],class_id)

    # Get segmentation map of target class (T/F)
    print('colormap: ',colormap[target_class])
    seg_color = (colormap[target_class][2], colormap[target_class][1], colormap[target_class][0])
    seg_map = np.all(output == seg_color, axis=-1)
```

- [x] 코드가 간결한가요?

- 함수로 구현하여 코드반복을 줄이고, 깔끔하게 만들었다.

```python
#chromakey function
def apply_chromakey(_img_path, _target_class, _chroma_path):
    img = cv2.imread(_img_path)
    target_class = _target_class
    chroma_img = cv2.imread(chroma_path)
    chroma_img_resize = cv2.resize(chroma_img, (img.shape[1],img.shape[0]))
    
    segvalues, output = model.segmentAsPascalvoc(_img_path)

    # Get segmentation map of target class (255/0)
    seg_color = (colormap[target_class][2], colormap[target_class][1], colormap[target_class][0])
    seg_map = np.all(output == seg_color, axis=-1)
    img_mask = seg_map.astype(np.uint8) * 255
    img_mask_bgr = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

    img_concat_chroma = np.where(img_mask_bgr==255, img, chroma_img_resize)
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img_concat_chroma, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return img_concat_chroma
    ```
