## Code Peer Review Template
---
* 코더 : 정형준
* 리뷰어 : 김창완


## PRT(PeerReviewTemplate)
---
- 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? o
```python
model_bi_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_bi_lstm = model_bi_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[es_bi_lstm, mc_bi_lstm])
test_loss_bi_lstm, test_accuracy_bi_lstm = model_bi_lstm.evaluate(X_test, y_test, verbose=0)
print("Bidirectional LSTM Test Accuracy:", test_accuracy_bi_lstm)
#결과
Bidirectional LSTM Test Accuracy: 0.8539780974388123
```
- 성공적으로 85%이상 accuracy가 나왔습니다
- 시각화도 문제없이 됩니다
- 모델도 1D-CNN, LSTM, Bidiraectioinal LSTM의 3가지 모델을 학습하였습니다.

-  주석을 보고 작성자의 코드가 이해되었나요? X  
	- 대부분의 코드에 주석이 없었습니다
-  코드가 에러를 유발할 가능성이 있나요?  O
	- warning이 많아 조금 위험하긴 합니다
```python
WARNING:absl:Found untraced functions such as lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: best_model.lstm/assets
INFO:tensorflow:Assets written to: best_model.lstm/assets
Epoch 2/10
3655/3655 [==============================] - 26s 7ms/step - loss: 0.6932 - acc: 0.4974 - val_loss: 0.6931 - val_acc: 0.5024

Epoch 00002: val_acc improved from 0.49761 to 0.50239, saving model to best_model.lstm
WARNING:absl:Found untraced functions such as lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses, lstm_cell_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: best_model.lstm/assets
INFO:tensorflow:Assets written to: best_model.lstm/assets
```
	
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)  
	- 아쉽게도 경력이 길지는 않으셔서 그런지 코드를 이해하지는 못하셨습니다. 다만 GPT를 활용해 어떻게든 정답을 찾으시려는 노력은 있었습니다.
```python
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
print(X_train.shape)
```
	- 이 구간에서 pre로 패딩을 했다면 어땠을까 아쉬움이 듭니다

- [x] 코드가 간결한가요?  
	- 코드 자체는 간결했습니다
