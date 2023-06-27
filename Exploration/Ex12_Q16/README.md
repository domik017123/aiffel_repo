# Code Peer Review Templete
- 코더 : 이동익
- 리뷰어 : 김다인


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  -정상적으로 동작합니다
  -데이터 전처리와 모델 설계, 학습이 잘 진행되었습니다
  -실제 결과와 요약문 비교는 안되었지만 결과와 요약이 구분되어 이해할 수 있었습니다
  -Summa을 이용해서 추출적 요약이 잘 되었습니다
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
  -각 코드별로 설명이 필요한 부분에 주석이 잘 들어가있었습니다
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
- [⭕] 5.코드가 간결한가요?
  -간결합니다

# 예시
```
# 데이터 분포가 최대길이에 몰려있어 조정하지 않고 그대로 사용하기로 함
text_max_len = 66
headlines_max_len = 16

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s.split()) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))

below_threshold_len(text_max_len, data['text'])
below_threshold_len(headlines_max_len,  data['headlines'])
```
데이터를 정제할 때 이유를 같이 설명해주셔서 이해하기 좋았습니다
