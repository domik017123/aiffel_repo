# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이동익
- 리뷰어 : 김재환


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [O] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
네, 모든 코드가 정상적으로 동작합니다.  
- 주어진 문제였던 `Augmentation 방법 3가지 (resize, cropping, mirroring)`를 활용해 데이터 증강을 수행했습니다.  
- U-Net을 class를 활용해 구현하고 그래프로 잘 표현했습니다. `Skip connection 구현` 부분을 첨부합니다.

```Python
class UNetGenerator(Model):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        encode_filters = [64,128,256,512,512,512,512,512]
        decode_filters = [512,512,512,512,256,128,64]
        
        self.encode_blocks = []
        for i, f in enumerate(encode_filters):
            if i == 0:
                self.encode_blocks.append(EncodeBlock(f, use_bn=False))
            else:
                self.encode_blocks.append(EncodeBlock(f))
        
        self.decode_blocks = []
        for i, f in enumerate(decode_filters):
            if i < 3:
                self.decode_blocks.append(DecodeBlock(f))
            else:
                self.decode_blocks.append(DecodeBlock(f, dropout=False))
        
        self.last_conv = layers.Conv2DTranspose(3, 4, 2, "same", use_bias=False)
    
    def call(self, x):
        #skip connection을 위한 리스트에 Encoder 출력 할당
        features = []
        for block in self.encode_blocks:
            x = block(x)
            features.append(x)
            
        #Encoder의 마지막 출력은 features에서 제외
        features = features[:-1]
        
        #features의 역순으로 연결            
        for block, feat in zip(self.decode_blocks, features[::-1]):
            x = block(x)
            x = layers.Concatenate()([x, feat])
        
        x = self.last_conv(x)
        return x
                
    def get_summary(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
    
    def get_plot(self, input_shape=(256,256,3)):
        inputs = Input(input_shape)
        model = Model(inputs, self.call(inputs))
        return tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

UNetGenerator().get_plot()
```
- Segments, real_image, predicted image를 잘 비교했습니다. 
- generator와 discriminator 각각 loss의 변화를 그래프로 출력했습니다.  

- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?
- 네, markdown으로 각 단계를 잘 구분하였고, 특히 구현이 어려웠던 `학습 단계별 가중치 업데이트` 과정의 주석이 이해를 도왔습니다.
```Python
def train_step(input_image, real_image):
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
        # 생성자에 입력 스케치를 전달하여 가짜 컬러 이미지 생성
        fake_image = generator(input_image, training=True)

        # 판별자에 실제 컬러 이미지와 가짜 컬러 이미지를 전달하여 판별 결과 계산
        real_disc = discriminator(real_image, input_image, training=True)
        fake_disc = discriminator(fake_image, input_image, training=True)
        
        #Generator loss + lambda * L1 loss (lambda = 100)
        gene_loss, l1_loss = get_gene_loss(fake_image, real_image, fake_disc)
        gene_total_loss = gene_loss + 100*l1_loss
        
        #Discriminator loss
        disc_loss = get_disc_loss(fake_disc, real_disc)
        

    # 생성자와 판별자에 대한 그래디언트 계산
    gene_gradient = gene_tape.gradient(gene_total_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
```
- [X] 3.코드가 에러를 유발할 가능성이 있나요?

- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
- 네, 구분이 크게 필요하지 않은 부분은 공백을 최대한 줄이되 구분이 필요한 부분은 공백을 잘 활용했습니다. 

- [O] 5.코드가 간결한가요?
- 네, 실행할 코드 블록이 총 18개로 많지 않습니다. 

# 참고 링크 및 코드 개선
(PEP) `함수 정의`와 `실행`을 같은 코드블럭에서 할 경우 함수 정의 후 `Enter 2번 입력` 후 실행합니다.
(경험) PyCharm으로 코드 작성 시 출력되는 문법 지적(?)입니다.  


```Python 
import matplotlib.pyplot as plt
def imshow_compare(original, contrast):
    plt.subplot(121)
    plt.imshow(original)

    plt.subplot(122)
    plt.imshow(contrast)

    plt.show()
# 엔터를
# 두 번 입력합니다.
imshow_compare(ori, con)
```
