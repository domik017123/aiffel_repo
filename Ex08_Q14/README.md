## Code Peer Review Template
---
* 코더 : 이동익
* 리뷰어 : 정연준


## PRT(PeerReviewTemplate)
---
- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```python
test_loss_lstm, test_accuracy_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)
print("LSTM Test Accuracy:", test_accuracy_lstm)

test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)
print("1D-CNN Test Accuracy:", test_accuracy_cnn)

test_loss_bi_lstm, test_accuracy_bi_lstm = model_bi_lstm.evaluate(X_test, y_test, verbose=0)
print("Bidirectional LSTM Test Accuracy:", test_accuracy_bi_lstm)
```
- 3가지 모델이 성공적으로 시도하어 Text Classification을 성공적으로 구현하였음

- [x] 주석을 보고 작성자의 코드가 이해되었나요?

- 코드 주석과 내용, 결과를 보면 코드가 이해가 충분히 되었습니다.
- 
- [x] 코드가 에러를 유발할 가능성이 있나요?
```
# 모델 구성
model_bi_lstm = Sequential(name='Bidirectional_LSTM')
model_bi_lstm.add(Embedding(num_words, word_vector_dim, name='emb'))
model_bi_lstm.add(Bidirectional(LSTM(64, name='bi_lstm')))
model_bi_lstm.add(Dense(64, activation='relu', name='d_rel'))
model_bi_lstm.add(Dense(1, activation='sigmoid', name='d_sig'))

model.summary()
```
- 임베딩시에 초기화가 word2vec으로 되지않아 학습이 진행되었다. 초기화하여 학습을진행해주기만 하면 충분할 것같다.

- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
```python
es_bi_lstm = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc_bi_lstm = ModelCheckpoint('best_model.bi_lstm', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model_bi_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_bi_lstm = model_bi_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[es_bi_lstm, mc_bi_lstm])
```
- 원활한 학습을 위해 EarlyStoping 기능과 ModelCheckpoint 기능을 구현하였음

- [x] 코드가 간결한가요?

- 코드 진행 상황 중 체크하고 싶은건 다 보입니다.
