# AIFFEL Campus Online 4th Code Peer Review
- 코더 : 이동익
- 리뷰어 : 김설아


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
def add_sticker(img, boxes, box_index, img_sticker):
    img_show = img.copy()
#     print(img_show.shape)
    img_height = img_show.shape[0]
    img_width = img_show.shape[1]

    x_min = int(boxes[box_index][0] * img_width)
    y_min = int(boxes[box_index][1] * img_height)
    x_max = int(boxes[box_index][2] * img_width)
    y_max = int(boxes[box_index][3] * img_height)
#     print(x_min,x_max,y_min,y_max)

    # 스티커 얼굴 크기로 resize
    w = x_max - x_min
    w_sticker = w
    h_sticker = w
    img_sticker = cv2.resize(img_sticker, (w_sticker,h_sticker))
#     print(img_sticker.shape)

    # 스티커 좌표
    refined_x = x_min
    refined_y = y_min - 4*h_sticker//5

    # 스티커가 영역 밖으로 나가는 경우 crop
    if refined_x < 0: 
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:, :]
        refined_y = 0
    
    # 스티커 적용
    sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1],:] 

    img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x + img_sticker.shape[1],:] \
    = cv2.addWeighted(sticker_area, 1, img_sticker, 2, 0)
        
    return img_show
 ```
 > 세세한 주석으로 이해가 수월했습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
 ```python
class PiecewiseConstantWarmUpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, boundaries, values, warmup_steps, min_lr, name=None):
        super(PiecewiseConstantWarmUpDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
                    "The length of boundaries should be 1 less than the"
                    "length of values")

        self.boundaries = boundaries
        self.values = values
        self.name = name
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        with tf.name_scope(self.name or "PiecewiseConstantWarmUp"):
            step = tf.cast(tf.convert_to_tensor(step), tf.float32)
            pred_fn_pairs = []
            warmup_steps = self.warmup_steps
            boundaries = self.boundaries
            values = self.values
            min_lr = self.min_lr

            pred_fn_pairs.append(
                (step <= warmup_steps,
                 lambda: min_lr + step * (values[0] - min_lr) / warmup_steps))
            pred_fn_pairs.append(
                (tf.logical_and(step <= boundaries[0],
                                step > warmup_steps),
                 lambda: tf.constant(values[0])))
            pred_fn_pairs.append(
                (step > boundaries[-1], lambda: tf.constant(values[-1])))

            for low, high, v in zip(boundaries[:-1], boundaries[1:],
                                    values[1:-1]):
                pred = (step > low) & (step <= high)
                pred_fn_pairs.append((pred, lambda: tf.constant(v)))

            return tf.case(pred_fn_pairs, lambda: tf.constant(values[0]),
                           exclusive=True)

def MultiStepWarmUpLR(initial_learning_rate, lr_steps, lr_rate,
                      warmup_steps=0., min_lr=0.,
                      name='MultiStepWarmUpLR'):
    assert warmup_steps <= lr_steps[0]
    assert min_lr <= initial_learning_rate
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)
    return PiecewiseConstantWarmUpDecay(
        boundaries=lr_steps, values=lr_steps_value, warmup_steps=warmup_steps,
        min_lr=min_lr)
 ```
 > 클래스를 활용해 정리를 하여 에러 유발 가능성을 배제하셨습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
FILE_PATH = os.path.join(PROJECT_PATH, 'weights', 'weights_epoch_100.h5')
TEST_IMAGE_PATH = os.path.join(PROJECT_PATH, 'image_people.png')
STICKER_PATH = os.path.join(PROJECT_PATH, 'king.png')

def inference_test(filepath, test_image_path, sticker_path, is_bbox=False, is_sticker=False):
    model.load_weights(filepath)

    img_raw = cv2.imread(test_image_path)
    img_raw = cv2.resize(img_raw, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.float32(img_raw.copy())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad_params = pad_input_image(img, max_steps=max(BOX_STEPS))
    img = img / 255.0

    boxes = default_box()
    boxes = tf.cast(boxes, tf.float32)

    predictions = model.predict(img[np.newaxis, ...])

    pred_boxes, labels, scores = parse_predict(predictions, boxes)
    pred_boxes = recover_pad(pred_boxes, pad_params)

    if is_bbox == True:
        for box_index in range(len(pred_boxes)):
            img_raw = draw_box_on_face(img_raw, pred_boxes, labels, scores, box_index, IMAGE_LABELS)
    
    if is_sticker == True:
        img_sticker = cv2.imread(sticker_path)
        for box_index in range(len(pred_boxes)):
            img_raw = add_sticker(img_raw, pred_boxes, box_index, img_sticker)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.show()
 ```
 > 코드를 이해하고 스티커를 4명의 사람이 있는 사진의 전원에게 붙이는 과제를 성공하셨습니다.

- [x] **5. 코드가 간결한가요?**  
  
 ```python
inference_test(FILE_PATH, TEST_IMAGE_PATH, STICKER_PATH, is_bbox=False, is_sticker=True)

 ```
 > 함수로 정리된 간결한 코드를 볼 수 있었습니다.
