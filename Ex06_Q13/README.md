# AIFFEL_Project

<aside>
🔑 **PRT(Peer Review Template)**

  - [o]  1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요? <br>
  정상적으로 동작하고, 주어진 문제들도 다 해결하였습니다.
  #### Step 1. 스티커 구하기 or 만들기
  <pre><cpde>
  my_image_path =os.getenv('HOME') + '/aiffel/camera_sticker/images/selfie_1.jpg'
  img_bgr = cv2.imread(my_image_path) #opencv >> bgr
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  </code></pre>
  #### Step 2. 얼굴 검출 & 랜드마크 검출 하기
  <pre><cpde>
  face_detector = dlib.get_frontal_face_detector()
  dlib_rects = face_detector(img_rgb, 1) #dib >> rgb 
  print(dlib_rects)
  
  img_show = img_bgr.copy()

  for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

    img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img_show_rgb)
    plt.show()
  </code></pre>
   <pre><code>
    model_path = os.getenv('HOME') + '/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
    landmark_predictor = dlib.shape_predictor(model_path)

    list_landmarks = []

    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)

    for landmark in list_landmarks:
        for point in landmark:
            cv2.circle(img_show, point, 2, (0, 255, 255), -1)

    img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    plt.imshow(img_show_rgb)
    plt.show()
    </code></pre>
  #### Step 3. 스티커 적용 위치 확인하기
    <pre><code>
    x = landmark[30][0]
    y = landmark[30][1]
    w = h = dlib_rect.width()
    w_sticker = w-w//5
    h_sticker = h-h//5
    img_sticker = cv2.resize(img_sticker, (w_sticker,h_sticker))
    print (f'nose(x,y) : ({x},{y})')
    print (f'dlib_rect(w,h) : ({w},{h})')
    print (f'sticker_size(w,h) : ({len(img_sticker[0])},{len(img_sticker[1])})')

    refined_x = x - w_sticker//2
    refined_y = y - h_sticker//2

    #crop minus location
    if refined_x < 0: 
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:, :]
        refined_y = 0
    print (f'sticker_crop(x,y) : ({refined_x},{refined_y})')   
    </code></pre>
    #### Step 4. 스티커 적용하기
    <pre><code>
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
    </code></pre>
  #### Step 5. 문제점 찾아보기 <br>
  (1) 멀리서 촬영된 경우<br>
  (2) 얼굴이 기울어진 경우<br>
  (3) 어두운 사진의 경우<br>
  에 대한 test를 진행하였고, 2번 사항에 대한 해결 소스까지 작성하였습니다. 
  <pre><code>
    def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    return result
  </code></pre>
- [o]  2.주석을 보고 작성자의 코드가 이해되었나요?<br>
  네. 주석을 참고해서 코드를 쉽게 이해할 수 있었습니다.
- []  3.코드가 에러를 유발할 가능성이 있나요?
- [o]  4.코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [o]  5.코드가 간결한가요? <br>
   비교적 간결하게 작성되었습니다. 
  
</aside>
