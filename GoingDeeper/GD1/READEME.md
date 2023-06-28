# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 이동익
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
전부 정상적으로 진행됩니다. 하지만 학습시간상 그만둔 형태로 보이네요 아쉬워요!!<br>
![image](https://github.com/domik017123/aiffel_repo/assets/65104209/746e42a4-df38-44f1-a7aa-ef04b97fa1c6)

### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
#resnet50
def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    return x
```
<br> 어느부분이 ResNet50인지 34인지 구분감있게 주석을 달아주셨습니다! 멋져요!

### 코드가 에러를 유발할 가능성이 있나요? (X)
대부분 적재적소에 코드가 쓰여서 에러를 유발할 사항은 없습니다.! 또한 conv layer를 함수화함으로써 더 직관적입니다.
```python
def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if(i == 0):
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def create_model():
    # K = 2
    inputs = layers.Input(shape=(224, 224, 3), dtype='float32')
    # x = data_preprocess(inputs)
    x = conv1_layer(inputs)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)
    x = GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
```


### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
![image](https://github.com/domik017123/aiffel_repo/assets/65104209/5e3cf088-3d75-4bf6-81f6-f24156f88f29)

<br>skip connection에 대해 유연하게 사용한 부분이 멋지네요!!
### 코드가 간결한가요? (O)
위에서 말한대로 코드의 재사용을 위해 함수화한 부분이 간결합니다! 사용성도 높습니다 GOOD!
<br> 배우고갑니다 ꉂꉂ(ᵔᗜᵔ*)

----------------------------------------------
## 1 2 4 5 PASS!
