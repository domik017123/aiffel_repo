# AIFFEL_E_Project

<aside>
🔑 **PRT(Peer Review Template)**

- [o]  1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
       코드는 정상적으로 동작하고, 주어신 문제들도 해결 하였습니다. 
       다양한 앙상블 모델에 Parameter grid를 설정하고 for을 이용해 자동으로 학습할 수 있도록 설계하였습니다.  
<pre><code>
# Define the parameter grids for each model
param_grid_gboost = {
    'n_estimators': [50, 100],
    'max_depth': [1, 10],
}

param_grid_xgboost = {
    'n_estimators': [50, 100],
    'max_depth': [1, 10],
}

param_grid_lightgbm = {
    'n_estimators': [50, 100],
    'max_depth': [1, 10],
}

param_grid_rdforest = {
    'n_estimators': [50, 100],
    'max_depth': [1, 10],
}

# Perform grid search for each model
grid_results = {}
best_params = {}

for model in models:
    model_name = model.__class__.__name__
    
    if model_name == 'GradientBoostingRegressor':
        params = param_grid_gboost
    elif model_name == 'XGBRegressor':
        params = param_grid_xgboost
    elif model_name == 'LGBMRegressor':
        params = param_grid_lightgbm
    elif model_name == 'RandomForestRegressor':
        params = param_grid_rdforest
    
    results = my_GridSearch(model, x, y, params, verbose=2, n_jobs=5)
    best_params[model_name] = results.iloc[0].to_dict()
    grid_results[model_name] = results

# Create the average blending model with the best parameters
models_with_best_params = []
weights = []

for model_name, params in best_params.items():
    model = models[model_name]
    model.set_params(**params)
    models_with_best_params.append((model_name, model))
    weights.append(1)  # Equal weights for each model

blending_model = VotingRegressor(estimators=models_with_best_params, weights=weights)

# Fit the blending model
blending_model.fit(x, y)
sub_pred = blending_model.predict(sub)
</code></pre>

- []  2.주석을 보고 작성자의 코드가 이해되었나요?
  이해가 쉽도록 잘 정리되었습니다. 
- []  3.코드가 에러를 유발할 가능성이 있나요?
- []  4.코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- []  5.코드가 간결한가요?
  코드는 간결합니다. 
</aside>
