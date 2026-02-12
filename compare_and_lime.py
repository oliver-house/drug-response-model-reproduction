"""
Core evaluation and explainability functions for the panobinostat reproduction study.
Implements ridge performance comparison under provided splits or repeated stratified
cross-validation, and LIME-based local explanations for selected test cases.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from lime.lime_tabular import LimeTabularExplainer

def compare_panobinostat(data, 
                         model,
                         mode='provided',
                         seed=0,
                         cv_n_splits=5,
                         cv_n_repeats=10,
                         cv_y_bins=10,
):
    """
    Evaluate ridge performance for EC11K and MC9K using either provided splits or repeated stratified cross-validation, 
    returning per-split metrics, summary statistics, and the mean R² difference.
    """
    rows = []
    summaries = {}
    for label in data:
        r2_list, rmse_list = [], []
        if mode == 'provided':
            for i in range(1, 4):
                sub = data[label]['splits'][i]
                X_train = sub['X_train']
                y_train = sub['y_train']
                X_test = sub['X_test']
                y_test = sub['y_test']
                m = clone(model)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rows.append({'dataset': label, 'split': i, 'R2': r2, 'RMSE': rmse})
                r2_list.append(r2)
                rmse_list.append(rmse)
        elif mode == 'cv':
            X = data[label]['X']
            y = data[label]['y'].reshape(-1)
            y = np.asarray(y, dtype=float)
            bins = int(cv_y_bins)
            if bins < 2:
                bins = 2
            qs = np.linspace(0.0, 1.0, bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
            if edges.size < 3:
                edges = np.unique(np.quantile(y, [0.0, 0.5, 1.0]))
            y_bins = np.digitize(y, edges[1:-1], right=True)
            rskf = RepeatedStratifiedKFold(
                n_splits=int(cv_n_splits),
                n_repeats=int(cv_n_repeats),
                random_state=int(seed),
            )
            fold_id = 0
            for train_idx, test_idx in rskf.split(X, y_bins):
                fold_id += 1
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]
                m = clone(model)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rows.append({'dataset': label, 'split': fold_id, 'R2': r2, 'RMSE': rmse})
                r2_list.append(r2)
                rmse_list.append(rmse)
        else:
            raise ValueError("mode must be 'provided' or 'cv'")
        
        r2s = np.array(r2_list, dtype=float)
        rmses = np.array(rmse_list, dtype=float)
        summaries[label] = {
            'R2_mean': float(r2s.mean()), 
            'R2_sd': float(r2s.std(ddof=1)), 
            'RMSE_mean': float(rmses.mean()), 
            'RMSE_sd': float(rmses.std(ddof=1)),
        }
    delta_r2 = float(summaries['EC11K']['R2_mean'] - summaries['MC9K']['R2_mean'])
    return rows, summaries, delta_r2

def lime_panobinostat(
        data, 
        model, 
        num_cases=4, 
        num_features=10, 
        mode='regression', 
        label='EC11K', 
        split=1,
        seed=None,
):
    """
    Fit the model on a specified split and generate LIME explanations for the lowest predicted ln(IC50) test cases.
    """
    sub = data[label]['splits'][split]
    X_train = sub['X_train']
    y_train = sub['y_train']
    X_test = sub['X_test']
    y_test = sub['y_test']
    n_columns = X_train.shape[1]
    m = clone(model)
    m.fit(X_train, y_train)
    yhat = m.predict(X_test)
    explainer = LimeTabularExplainer(
        training_data=X_train, 
        feature_names=[f'gene_{j:05d}' for j in range(n_columns)], 
        mode=mode,
        random_state=seed,
    )
    cases = []
    for case_id, idx in enumerate(np.argsort(yhat)[:num_cases], start=1):
        x_i = X_test[idx]
        pred = float(yhat[idx])
        obs = float(y_test[idx])
        exp = explainer.explain_instance(
            data_row=x_i, 
            predict_fn=lambda a: m.predict(np.atleast_2d(a)), 
            num_features=num_features,
        )
        features = [{'feature': str(feat), 'weight': float(w)} for feat, w in exp.as_list()]
        cases.append({
            'case': case_id, 
            'test_row': int(idx), 
            'pred_lnIC50': pred, 
            'obs_lnIC50': obs, 
            'features': features,
        })
    final = {
        'dataset': label, 
        'split': split, 
        'num_cases': num_cases, 
        'num_features': num_features, 
        'cases': cases,
    }
    return final