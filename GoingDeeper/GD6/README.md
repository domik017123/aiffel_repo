# AIFFEL Campus Online 4th Code Peer Review
- 코더 : 이동익
- 리뷰어 : 김설아


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
import matplotlib.pyplot as plt
import keras_ocr

img_path_list = [HOME_DIR + '/captcha_test.jpg' ,HOME_DIR + '/captcha_test2.jpg'] 


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
images = [
    keras_ocr.tools.read(img_path) for img_path in img_path_list
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
 ```
 > 주석을 통해 어떤 과정에 해당하는 코드인지 이해하기 수월했습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
 ```python
tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
 ```
 > 시각화를 통해 모델 확인 과정을 거쳐 에러 유발 가능성을 없애셨습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
from IPython.display import display

# 모델이 inference한 결과를 글자로 바꿔주는 역할을 합니다
# 코드 하나하나를 이해하기는 조금 어려울 수 있습니다
def decode_predict_ctc(out, chars = TARGET_CHARACTERS):
    results = []
    indexes = K.get_value(
        K.ctc_decode(
            out, input_length=np.ones(out.shape[0]) * out.shape[1],
            greedy=False , beam_width=5, top_paths=1
        )[0][0]
    )[0]
    text = ""
    
    '''
    tf.keras.backend.ctc_decode :
        blank labels are returned as -1
        >> 이거 때문에 Result: SLINKING9999999999999999 이런식으로 출력됨
    '''
    for index in indexes:
        if index != -1: ## blank인 경우 출력X
            text += chars[index]
    results.append(text)
    return results

# 모델과 데이터셋이 주어지면 inference를 수행합니다
# index개 만큼의 데이터를 읽어 모델로 inference를 수행하고
# 결과를 디코딩해 출력해줍니다
def check_inference(model, dataset, index = 5):
    for i in range(index):
        inputs, outputs = dataset[i]
        img = dataset[i][0]['input_image'][0:1,:,:,:]
        output = model.predict(img)
        result = decode_predict_ctc(output, chars="-"+TARGET_CHARACTERS)[0].replace('-','')
        print("Result: \t", result)
        display(Image.fromarray(img[0].transpose(1,0,2).astype(np.uint8)))

check_inference(model_pred, test_set, index=10)
 ```
 > 오류의 원인을 찾고 코드를 수정하셨습니다.

- [x] **5. 코드가 간결한가요?**  
  
 ```python
import keras_ocr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HOME_DIR = os.getenv('HOME')
SAMPLE_IMG_PATH = HOME_DIR + '/data/sample.jpg'

def detect_text(img_path):
    img = Image.open(img_path)
    img_draw = ImageDraw.Draw(img)

    #keras-ocr의 입력 차원에 맞게 변경
    img_arr = np.asarray(img)
    img_arr = np.expand_dims(img_arr, axis=0)
#     print('IMG:',img_arr.shape)

    #Detector.detect
    """Recognize the text in a set of images.
    Args:
        images: Can be a list of numpy arrays of shape HxWx3 or a list of
            filepaths.
    """
    ocr_result = detector.detect(img_arr)
    ocr_result = np.squeeze(ocr_result)
#     print('Dectected boxes:\n',ocr_result)

    #Crop >> Recognition model input
#     print('Cropped boxes:')
    cropped_imgs = []
    for text_result in ocr_result:
        # Detector output
        img_draw.polygon(text_result, outline='red')
        # Crop
        x_min = text_result[:,0].min() - 5
        x_max = text_result[:,0].max() + 5
        y_min = text_result[:,1].min() - 5
        y_max = text_result[:,1].max() + 5
        word_box = [x_min, y_min, x_max, y_max]
#         print(word_box)
        img_draw.rectangle(word_box, outline='green')
        cropped_imgs.append(img.crop(word_box))

    plt.figure(figsize=(20,10))    
    plt.imshow(img)
    plt.show()
    
    return cropped_imgs


cropped_imgs = detect_text(SAMPLE_IMG_PATH)

 ```
 > 함수를 적절히 활용하여 간결한 코드가 될 수 있게 하셨습니다.

    
