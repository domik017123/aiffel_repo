

# Code Peer Review Templete
--------
- 코더 : 이동익
- 리뷰어 : 박재영
# PRT(Peer Review Templete)
--------
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요  

[O] 1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요  
[O] 2. 주석을 보고 작성자의 코드가 이해되었나요?  
[O] 3. 코드가 에러를 유발한 가능성이 있나요?  
[O] 4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?  
[O] 5. 코드가 간결한가요?  

# 참고링크 및 코드 개선 여부 
----------
### 1. 코드 블럭이 잘 구분되어 있고, 코드가 순차적으로 구성되어 있어 구조를 파악하기 쉬었습니다.

'''  
  
#sticker location : nose
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    x = landmark[30][0]
    y = landmark[30][1]
    w = h = dlib_rect.width()
    w_sticker = w-w//5
    h_sticker = h-h//5
    ....
    #crop minus location
    if refined_x < 0: 
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:, :]
        refined_y = 0
    print (f'sticker_crop(x,y) : ({refined_x},{refined_y})')
    ....
    ###transparency control### > same as addWeighted()
    sticker_transparency = 0.7
    #remove img's alpha channel in sticker_area
    sticker_area = img_bgra[refined_y:refined_y+img_sticker.shape[0] \
                            , refined_x:refined_x+img_sticker.shape[1], :3] 
    #split alpha and bgr channel of sticker > change alpha value(transparency)
    sticker_alpha = img_sticker[:, :, 3] * sticker_transparency / 255.0  # alpha(0~1)
    sticker_bgr = img_sticker[:, :, :3]  # bgr channel
    #merge img and sticker with changed transparency
    sticker_merged = sticker_alpha[:, :, None] * sticker_bgr + (1 - sticker_alpha[:, :, None]) * sticker_area
    #apply to img_show
    img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x + img_sticker.shape[1], :3] \
    = sticker_merged.astype(np.uint8)
    
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGRA2RGBA))
    plt.show()

'''
  1. x, y 좌표 및 w, h 값을 설정
  2. 스티커의 크기를 조정
  3. 투명도(Transparency)를 조절
  4. 이미지를 출력
 - 과정으로 프로세스 정의가 잘 되어 있습니다.
 
3. 주석을 보고 작성자의 코드가 이해되었나요? - 네
    - Bikesharing 페이지 각 블록 마다 간결하게 주석 처리가 되어 있었습니다.
4. 코드가 에러를 유발할 가능성이 있나요?
    - 없어보입니다.  
5. 코드가 간결한가요?
    - 코드 문법이 간결해서, 향후 리팩토링 하기 좋은 구조를 가지고 있습니다.
