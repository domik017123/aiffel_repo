# AIFFEL_E_Project

<aside>
🔑 **PRT(Peer Review Template)**

- []  1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- []  2.주석을 보고 작성자의 코드가 이해되었나요?
- []  3.코드가 에러를 유발할 가능성이 있나요?
- []  4.코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- []  5.코드가 간결한가요?
</aside>


<pre><code>

list_landmarks = []
# 랜드마크의 위치를 저장할 list 생성

# 얼굴 영역 박스 마다 face landmark를 찾아냅니다
# face landmark 좌표를 저장해둡니다
for dlib_rect in dlib_rects:
points = landmark_predictor(img_rgb, dlib_rect)
# 모든 landmark의 위치정보를 points 변수에 저장
list_points = list(map(lambda p: (p.x, p.y), points.parts()))
# 각각의 landmark 위치정보를 (x,y) 형태로 변환하여 list_points 리스트로 저장
list_landmarks.append(list_points)
# list_landmarks에 랜드마크 리스트를 저장

print(len(list_landmarks[0]))
# 얼굴이 n개인 경우 list_landmarks는 n개의 원소를 갖고
# 각 원소는 68개의 랜드마크 위치가 나열된 list

</code></pre>
