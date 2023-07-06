# AIFFEL Campus Online 4th Code Peer Review
- 코더 : 이동익
- 리뷰어 : 최지호


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [X] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- ![image](https://github.com/domik017123/aiffel_repo/assets/79844211/34e95cdb-d1b0-4918-829b-c6fe969c7807)
- 전부 만족하였습니다.
  
- [X] 2.주석을 보고 작성자의 코드가 이해되었나요?  
  - ![image](https://github.com/domik017123/aiffel_repo/assets/79844211/98f32a5a-e71d-489d-b6b8-abb389c3af0c)
  - dice가 공식만 보고는 정확히 어떤 개념인지 몰랐는데 동익님께서 참고 이미지까지 첨부 해주셔서 저도 코드를 이해하는데 도움이 되었습니다.

- [ ] 3.코드가 에러를 유발할 가능성이 있나요?
  - 아니오.
    
- [X] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  - ![image](https://github.com/domik017123/aiffel_repo/assets/79844211/14d560d6-b9fb-4cc2-a13c-d9df904dd238)
  - ```python
    #Convolution encoder: conv, batch, ReLU, maxpool(option)
    def encoder_block(x, fillter_n, max_pooling=False):
      enc = Conv2D(fillter_n, 3, padding='same')(x)
      enc = BatchNormalization()(enc)
      enc = tf.keras.activations.relu(enc)
      
      enc = Conv2D(fillter_n, 3, padding='same')(enc)
      enc = BatchNormalization()(enc)
      enc = tf.keras.activations.relu(enc)

      if max_pooling:
          pool = MaxPooling2D(pool_size=2)(enc)
          return enc, pool
      return enc

    def build_model2(input_shape=(224, 224, 3)):
        inputs = Input(input_shape)  
    
        #down-sampling
        x0_0, dx0_0 = encoder_block(inputs, 64, max_pooling=True)
        x1_0, dx1_0 = encoder_block(dx0_0, 128, max_pooling=True)
        x2_0, dx2_0 = encoder_block(dx1_0, 256, max_pooling=True)
        x3_0, dx3_0 = encoder_block(dx2_0, 512, max_pooling=True)
        x4_0 = encoder_block(dx3_0, 1024, max_pooling=False)
        
        #nested skip connection & up-sampling
        #L1 : x1_0 > x0_1 > output1
        ux1_0 = UpSampling2D(2)(x1_0)
        x0_1 = concatenate([x0_0, ux1_0], axis=3)
        x0_1 = encoder_block(x0_1, 64)
        
        output1 = Conv2D(1, 1, padding='same', activation='sigmoid')(x0_1)
        
        #L2 : x2_0 > x1_1 > x0_2 > output2
        ux2_0 = UpSampling2D(2)(x2_0)
        x1_1 = concatenate([x1_0, ux2_0], axis=3)
        x1_1 = encoder_block(x1_1, 128)
        
        ux1_1 = UpSampling2D(2)(x1_1)
        x0_2 = concatenate([x0_0, x0_1, ux1_1], axis=3)
        x0_2 = encoder_block(x0_2, 64)
        
        output2 = Conv2D(1, 1, padding='same', activation='sigmoid')(x0_2)
        
        #L3 : x3_0 > x2_1 > x1_2 > x0_3 > output3
        ux3_0 = UpSampling2D(2)(x3_0)
        x2_1 = concatenate([x2_0, ux3_0], axis=3)
        x2_1 = encoder_block(x2_1, 256)
        
        ux2_1 = UpSampling2D(2)(x2_1)
        x1_2 = concatenate([x1_0, x1_1, ux2_1], axis=3)
        x1_2 = encoder_block(x1_2, 128)
        
        ux1_2 = UpSampling2D(2)(x1_2)
        cx0_3 = concatenate([x0_0, x0_1, x0_2, ux1_2], axis=3)
        x0_3 = encoder_block(cx0_3, 64)
        
        output3 = Conv2D(1, 1, padding='same', activation='sigmoid')(x0_3)
        
        #L4 : x4_0 > x3_1 > x2_2 > x1_3 > x0_4 > output4
        ux4_0 = UpSampling2D(2)(x4_0)
        cx3_1 = concatenate([x3_0, ux4_0], axis=3)
        x3_1 = encoder_block(cx3_1, 512)
        
        ux3_1 = UpSampling2D(2)(x3_1)
        cx2_2 = concatenate([x2_0, x2_1, ux3_1], axis=3)
        x2_2 = encoder_block(cx2_2, 256)
        
        ux2_2 = UpSampling2D(2)(x2_2)
        cx1_3 = concatenate([x1_0, x1_1, x1_2, ux2_2], axis=3)
        x1_3 = encoder_block(cx1_3, 128)
        
        ux1_3 = UpSampling2D(2)(x1_3)
        cx0_4 = concatenate([x0_0, x0_1, x0_2, x0_3, ux1_3], axis=3)
        x0_4 = encoder_block(cx0_4, 128)
        
        output4 = Conv2D(1, 1, padding='same', activation='sigmoid')(x0_4)
        
        #outputs  
        outputs = (output1 + output2 + output3 + output4) / 4
                
        model = tf.keras.Model(inputs=inputs, outputs=outputs) 
        return model
    
    model2 = build_model2()
    model2.summary()
    ```
  - 개념과 그림, 코드가 서로 맞물려서 제대로 이해하고 코드를 작성하신 것이 한 눈에 보였습니다.

- [X] 5.코드가 간결한가요?
  - ```python
    unet_score_list = []
    unetpp_score_list = []
    for i in range(1,6):
        print("\nU-Net")
        output, prediction, target = get_output(
             model, 
             test_preproc,
             image_path=dir_path + f'/image_2/00{str(i).zfill(4)}_10.png',
             output_path=dir_path + f'./result_{str(i).zfill(3)}.png',
             label_path=dir_path + f'/semantic/00{str(i).zfill(4)}_10.png'
         )
        unet_iou_score = calculate_iou_score(target, prediction)
        unet_score_list.append(unet_iou_score)
    
        print("\nU-Net++")
        output, prediction, target = get_output(
             model2, 
             test_preproc,
             image_path=dir_path + f'/image_2/00{str(i).zfill(4)}_10.png',
             output_path=dir_path + f'./result_{str(i).zfill(3)}.png',
             label_path=dir_path + f'/semantic/00{str(i).zfill(4)}_10.png'
         )
        unetpp_iou_score = calculate_iou_score(target, prediction)
        unetpp_score_list.append(unetpp_iou_score)
    print("U-Net:", np.mean(unet_score_list))
    print("U-Net++:", np.mean(unetpp_score_list))
  ```
  - 네.
  - 특히 결과를 표시하는 부분이 인상깊었는데, 이 부분을 반복문을 사용해서 간결하게 표현해주셨습니다.
![image](https://github.com/201710808/aiffel_repo/assets/79844211/5e5d1434-31ed-44c7-be49-18bb9afabab3)

