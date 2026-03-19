#######################################################################################
# ProgPerfMetrics.py
# This file contains functions to calculate and plot various prognostic performance metrics
# such as RMSE, S-score, SISFE, and Prognostic Horizon. It also includes functions to plot the true vs. predicted RUL,
# RMSE over time, and S-score time series. The metrics are designed to evaluate the performance of RUL 
# prediction models to support the analysis of prognostic performance in a comprehensive manner for the
# thesis project  Human-Machine Collaboration in Industry 5.0: Linking prognostic Algorithms to Sustainable, 
# Resilient, and Human-Centric Maintenance.
# ###########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Save metrics_data to a text file
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def compute_rmse(y_true, y_pred):
    """
    Root Mean Square Error (single aggregate value).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = np.mean((y_pred - y_true) ** 2)
    return float(np.sqrt(mse))

def running_rmse(y_true, y_pred):
    """
    Running aggregate RMSE array (RMSE up to each sample i).
    Useful to plot RMSE evolution over time/samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return np.array([])
    accum = np.cumsum((y_pred - y_true) ** 2)
    idx = np.arange(1, n + 1)
    return np.sqrt(accum / idx)

def phm08_score(y_true, y_pred):
    """
    Original PHM'08 scoring function (asymmetric exponential).
    For reference / comparison (unbounded).
    d = pred - true
    if d < 0: score = exp(-d/13) - 1   (early prediction)
    else:    score = exp(d/10) - 1     (late prediction)
    Sum over samples and return average (or total if you prefer).
    Reference: Saxena et al., PHM 2008 scoring function.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.mean(scores)), scores  # returns mean and per-sample array

def sisfe(y_true, y_pred, MUL=None, k_pos=1.0, k_neg=2.0, mid_pos=None, mid_neg=None):
    """
    Scaling Independent Scoring Function Error (SISFE) - bounded, asymmetric sigmoid.

    - y_true, y_pred : arrays
    - MUL : maximum useful life (useful to pick midpoints relative to the problem scale).
            If None, we set MUL = max(y_true) - min(y_true) (a simple heuristic).
    - k_pos : steepness for positive errors (over-prediction / late predictions). Larger -> harsher penalty.
    - k_neg : steepness for negative errors (under-prediction / early predictions). Usually smaller than k_pos.
    - mid_pos, mid_neg : optional midpoints where sigmoid crosses 0.5 for pos/neg branches.
                         If None, defaults relative to MUL (practical defaults from the literature may vary).

    Returns:
      mean_sisfe, per_sample_sisfe, sign_mask  (sign_mask is True where prediction was late (d>0))
    Notes:
      - SISFE values are in (0,1). Lower is better if you interpret as "error magnitude" (we keep them as 0..1).
      - The original paper (Baptista et al., 2024) suggests a bounded, configurable sigmoid to avoid unbounded scores.
        This implementation follows that idea and exposes parameters for domain-specific tuning. :contentReference[oaicite:3]{index=3}

        Obs:
        # Choose MUL = max useful life for scaling (domain dependent). Here we use a heuristic:
        MUL = float(np.max(true_RUL_data))  # or pass a known maximum useful life
    """
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true  # positive -> late (over-pred), negative -> early (under-pred)
    n = len(d)
    if n == 0:
        return 0.0, np.array([]), np.array([])

    # Heuristic MUL if not provided (user should ideally set MUL to domain maximum useful life)
    if MUL is None:
        MUL = float(np.max(y_true) - np.min(y_true)) if len(y_true) > 1 else float(np.abs(y_true).max() if y_true.size else 1.0)
        if MUL == 0:
            MUL = 1.0

    # midpoints: where the sigmoid yields ~0.5 for each branch.
    # The paper offers recommendations to set midpoints relative to MUL; we follow the spirit and choose defaults.
    if mid_pos is None:
        mid_pos = 0.5 * MUL  # midpoint for late predictions (tunable)
    if mid_neg is None:
        mid_neg = 0.25 * MUL  # midpoint for early predictions (tunable)

    # sigmoid function that maps errors to (0,1): S(x) = 1 / (1 + exp(-k*(x - mid)))
    # Because we want a monotonic mapping of |d| to a bounded score, apply it separately to pos/neg sides.
    sisfe_vals = np.zeros_like(d, dtype=float)
    pos_mask = d > 0
    neg_mask = ~pos_mask

    # For late predictions (pos): use positive d
    if np.any(pos_mask):
        x_pos = d[pos_mask]
        sisfe_vals[pos_mask] = 1.0 / (1.0 + np.exp(-k_pos * (x_pos - mid_pos)))
    # For early predictions (neg): use absolute value of d but you may prefer to keep sign in offset
    if np.any(neg_mask):
        x_neg = -d[neg_mask]  # magnitude of early prediction (positive)
        sisfe_vals[neg_mask] = 1.0 / (1.0 + np.exp(-k_neg * (x_neg - mid_neg)))

    # Optionally shift/scale to be in [0,1) excluding exact 1.0 (numerical reasons)
    eps = 1e-12
    sisfe_vals = np.clip(sisfe_vals, 0.0, 1.0 - eps)

    mean_sisfe = float(np.mean(sisfe_vals))

    # 3. Print the quantitative result
    print(f"\n--- SISFE Result ---")
    print(f"SISFE CNN: {mean_sisfe:.2f}")
    print(f"SISFE ANN: {mean_sisfe:.2f} units")
    print(f"PH LSTM: {mean_sisfe:.2f} units")
    print("---------------------------------------")

    return mean_sisfe, sisfe_vals, pos_mask

"""
Calculates SISFE (Scaling Independent Scoring Function Error) - bounded, asymmetric sigmoid.
New version with configurable parameters for domain-specific tuning, following the spirit of 
Baptista et al., 2024. :contentReference[oaicite:3]{index=3}
"""
def calculateSISFE(y_true, y_pred):
    n_samples = len(y_true)
    sisfe_values = np.zeros(n_samples)
    sisfe_late_pred = []
    sisfe_early_pred = []
    sisfe_error = []
    late_idx = []
    early_idx = []

    # SISFE constants
    MUL = np.max(y_true)  # Maximum RUL value in the dataset
    c1 = 1.0 #0.5              # determine the steepness of the sigmoid functions for late predictions. Higher values make the function steeper.
    e1 = 2.0 #0.12          # determine the steepness of the exponential functions for early predictions (c1 > e1)
    e2 = 0.25 * MUL #1/2 * MUL
    #c2 = 1/3 * e2         # indicates the x-axis point where the function reaches 0.5. Maximum acceptable error ia 1
    c2 = 0.5 * MUL #1/625 * MUL 
    
    # Calculate SISFE and its time-series values
    sisfe_values = np.zeros(n_samples)
    for i in range(n_samples):
        error = y_pred[i] - y_true[i]
        if error > 0:  # Late prediction (undesirable)
            score = 1/(1 + np.exp(-c1 * (error - c2)))
            sisfe_values[i] = score
            sisfe_late_pred.append(score)  
            sisfe_error.append(error)
            late_idx.append(i)
        else:  # Early prediction (less severe)
            score = 1/(1 + np.exp(-e1 * (abs(error) - e2)))
            sisfe_values[i] = score
            sisfe_early_pred.append(score)  
            sisfe_error.append(error)
            early_idx.append(i)
    sisfe_score_mean = np.mean(sisfe_values)
    sisfe_score = sisfe_score_mean * sisfe_values.size

    return sisfe_score, sisfe_values, sisfe_error, sisfe_late_pred, sisfe_early_pred, late_idx, early_idx


def calculate_all_metrics(y_true, y_pred, a1=10, a2=13, regressor_type='CNN'):
    """
    Calculates various prognostic performance metrics.

    Args:
        y_true (np.array): Array of true RUL values.
        y_pred (np.array): Array of predicted RUL values.
        a1 (float): Penalty constant for late predictions (e_i > 0) for the S-score.
        a2 (float): Penalty constant for early predictions (e_i < 0) for the S-score.

    Returns:
        dict: A dictionary containing the calculated metrics and their time-series data.
    """
    
    n_samples = len(y_true)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Calculate S-score and its time-series values
    s_scores = np.zeros(n_samples)
    late_predictions = []
    late_idx = []
    early_predictions = []
    early_idx = []
    delta_late = []
    delta_early = []
    for i in range(n_samples):
        error = y_pred[i] - y_true[i]
        if error >= 0:  # Late prediction (undesirable)
            score = np.exp(error / a1) - 1
            s_scores[i] = score
            late_predictions.append(score)  
            late_idx.append(i)
            delta_late.append(error)
        else:  # Early prediction (less severe)
            score = np.exp(-error / a2) - 1
            s_scores[i] = score
            early_predictions.append(score)
            early_idx.append(i)
            #s_scores[i] = np.exp(-error / a2) - 1
            delta_early.append(error)
    s_score_mean = np.mean(s_scores)
    s_score = np.sum(s_scores)
    
    # Calculate SISFE and its time-series values
    sisfe_score, sisfe_values, sisfe_error, sisfe_late_pred, sisfe_early_pred, sisfe_late_idx, sisfe_early_idx = calculateSISFE(y_true, y_pred)
    
    return {
        'reg_type': regressor_type,
        'RMSE': rmse,
        'Delta_late': delta_late,
        'Delta_early': delta_early,
        'S_score_mean': s_score_mean,
        'S_score': s_score,
        'S_scores_time_series': s_scores,
        'late_pred': late_predictions,
        'late_idx': late_idx,
        'early_pred': early_predictions,
        'early_idx': early_idx,
        'SISFE': sisfe_score,
        'SISFE_time_series': sisfe_values,
        'SISFE_errors': sisfe_error,
        'SISFE_late_pred': sisfe_late_pred,
        'SISFE_early_pred': sisfe_early_pred,
        'SISFE_early_idx': sisfe_early_idx,
        'SISFE_late_idx': sisfe_late_idx,
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_rul_prediction(y_true, y_pred, rul_data=None, dataset=None):
    # Create x-axis ranges according to engine units numbers
    x_range = np.arange(1, len(y_true)+1)
    
    # x_range_N = len(x_range)
    # x2_range = np.arange(x_range_N)

    # if rul_data is not None:
    #     x_ANN_range = np.arange(1, len(rul_data['y_pred_ANN'])+1)
    #     x_LSTM_range = np.arange(1, len(rul_data['y_pred_LSTM'])+1)

    """Plots the true and predicted RUL over time."""
    #plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(x_range, y_true, label='True RUL', color='blue', linestyle='-')
    plt.plot(x_range, y_pred, label='Estimated RUL - CNN', color='red', linestyle='--', )
    #plt.scatter(x_range, y_pred, label='Estimated RUL - CNN', color='brown', )# marker='.', s=20
    if rul_data is not None:
        plt.plot(x_range, rul_data['y_pred_ANN'], label='Estimated RUL - ANN', color='g', linestyle='--') # marker='o', markersize=1,
        plt.plot(x_range, rul_data['y_pred_LSTM'], label='Estimated RUL - LSTM', color='m', linestyle='--')
    #plt.title('Prognostic Performance: True vs. Predicted RUL 📈')
    plt.xlabel('Unit Number: RUL in descending order')
    plt.ylabel('Remaining Useful Life (RUL)')
    
    
    ###################plt.xticks(np.arange(np.min(x_range), np.max(x_range), n_ticks))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = '1'
    ax.set_xticklabels(labels)
    plt.legend()
    plt.grid(True)
    plt.show()
    fig_path = './images/' + dataset + '_RUL.png'
    plt.savefig(fig_path)

def plot_rmse(y_true, y_pred, rul_data_dict=None, dataset=None):
    """
    Plots the squared error over time to visualize the RMSE.
    A single RMSE value is the mean of these squared errors.
    """
    # Calculate RMSE for ANN and LSTM if provided
    cnn_squared_error = (y_true - y_pred)**2
    cnn_rmse = np.sqrt(np.mean(cnn_squared_error))
    #cnn_squared_error = np.log(cnn_squared_error)

    if rul_data_dict is not None:
        ann_squared_error = (y_true - np.array(rul_data_dict['y_pred_ANN']))**2
        ann_rmse = np.sqrt(np.mean(ann_squared_error))
        lstm_squared_error = (y_true - np.array(rul_data_dict['y_pred_LSTM']))**2
        lstm_rmse = np.sqrt(np.mean(lstm_squared_error))
       
    plt.figure(figsize=(8, 4))
    plt.plot(cnn_squared_error, label='CNN', color='red')
    if rul_data_dict is not None:
        plt.plot(ann_squared_error, label='ANN', color='g')
        plt.plot(lstm_squared_error, label='LSTM', color='m')

    #plt.axhline(y=np.mean(cnn_squared_error), color='gray', linestyle='--', label=f'Mean Squared Error - CNN: {np.mean(cnn_squared_error):.2f}')
    #plt.axhline(y=cnn_rmse, color='red', linestyle='--', label=f'Root Mean Squared Error - CNN: {cnn_rmse:.2f}')
    # if rul_data_dict is not None:
    #     plt.axhline(y=ann_rmse, color='g', linestyle='--', label=f'Root Mean Squared Error - ANN: {ann_rmse:.2f}')
    #     plt.axhline(y=lstm_rmse, color='m', linestyle='--', label=f'Root Mean Squared Error - LSTM: {lstm_rmse:.2f}')
    
    #plt.title(f'Root Mean Squared Error (RMSE): {cnn_rmse:.2f} 📊')
    #plt.yscale('log', base=10)  # Set y-axis to logarithmic scale
    plt.xlabel('Time units')
    plt.ylabel('Squared Error')

    # Setting y-axis range
    #plt.ylim(-20, 100)
    
    plt.legend()
    plt.grid(True)
    plt.show()
    fig_path = './images/' + dataset + '_RMSE.png'
    plt.savefig(fig_path)

    # Print the quantitative result
    print(f"\n--- RMSE Performance Result ---")
    print(f"CNN RUL Predictions RMSE: {cnn_rmse:.2f}")
    print(f"ANN RUL Predictions RMSE: {ann_rmse:.2f}")
    print(f"LSTMM RUL Predictions RMSE: {lstm_rmse:.2f}")
    print("---------------------------------------")

def plot_s_score(s_scores_time_series, late_predictions, late_idx, early_predictions, early_idx, s_score_mean, s_score, 
                 m_dict_ANN=None, m_dict_LSTM=None, dataset=None):
    """Plots the S-score time series."""
    plt.figure(figsize=(8, 4))
    # CNN S-score
    plt.plot(s_scores_time_series, color='red', label=f'S-score - CNN: {s_score:.2f}')
    plt.scatter(late_idx, late_predictions, marker='^', color='red', edgecolor='orange', facecolor=(1, 0, 0, 0))#, edgecolor='orange'
    plt.scatter(early_idx, early_predictions, marker='o', edgecolor='steelblue', facecolor=(1, 0, 0, 0))#, edgecolor='steelblue', color='red',
    #plt.axhline(y=s_score_mean, color='gray', linestyle='--', label=f'Mean S-score: {s_score_mean:.2f}')

    # ANN S-score
    plt.plot(m_dict_ANN['S_scores_time_series'], color='g', label=f'S-score - ANN: {m_dict_ANN['S_score']:.2f}')
    plt.scatter(m_dict_ANN['late_idx'], m_dict_ANN['late_pred'], marker='^', color='g', edgecolor='orange', facecolor=(1, 0, 0, 0))#, edgecolor='orange'
    plt.scatter(m_dict_ANN['early_idx'], m_dict_ANN['early_pred'], marker='o', edgecolor='steelblue', facecolor=(1, 0, 0, 0))

    # LSTM S-score
    plt.plot(m_dict_LSTM['S_scores_time_series'], color='m', label=f'S-score - LSTM: {m_dict_LSTM['S_score']:.2f}')
    plt.scatter(m_dict_LSTM['late_idx'], m_dict_LSTM['late_pred'], marker='^', color='m', label='Late predictions', edgecolor='orange', facecolor=(1, 0, 0, 0))#, edgecolor='orange'
    plt.scatter(m_dict_LSTM['early_idx'], m_dict_LSTM['early_pred'], marker='o', label='early predictions', edgecolor='steelblue', facecolor=(1, 0, 0, 0))
    
    plt.title('S-score Over Time 📉')
    plt.xlabel('Time units')
    plt.ylabel('S-score')
    plt.legend()
    plt.grid(True)
    plt.show()
    fig_path = './images/' + dataset + '_S-Score.png'
    plt.savefig(fig_path)

    # Print the quantitative result
    print(f"\n--- S-score Performance Result ---")
    print(f"CNN RUL Predictions S-score: {s_score:.2f}")
    print(f"CNN Early Predictions Percentage  : {len(early_predictions)/len(s_scores_time_series)*100:.2f}%")
    print(f"CNN Late Predictions Percentage  : {len(late_predictions)/len(s_scores_time_series)*100:.2f}%")

    print(f"ANN RUL Predictions S-score: {m_dict_ANN['S_score']:.2f}")
    print(f"ANN Early Predictions Percentage  : {len(m_dict_ANN['early_pred'])/len(m_dict_ANN['S_scores_time_series'])*100:.2f}%")
    print(f"ANN Late Predictions Percentage  : {len(m_dict_ANN['late_pred'])/len(m_dict_ANN['S_scores_time_series'])*100:.2f}%")

    print(f"LSTMM RUL Predictions S-score: {m_dict_LSTM['S_score']:.2f}")
    print(f"LSTM Early Predictions Percentage  : {len(m_dict_LSTM['early_pred'])/len(m_dict_LSTM['S_scores_time_series'])*100:.2f}%")
    print(f"LSTM Late Predictions Percentage  : {len(m_dict_LSTM['late_pred'])/len(m_dict_LSTM['S_scores_time_series'])*100:.2f}%")
    print("---------------------------------------")



def calc_plot_sisfe(rul_data_dict):
    # SISFE (modern, bounded)

    # Store SISFE results in a dictionary
    sisfe_dict = {}


    # Choose MUL = max useful life for scaling (domain dependent). Here we use a heuristic:
    true_rul = rul_data_dict['y_true_CNN']
    MUL = float(np.max(true_rul))  # or pass a known maximum useful life

    # Compute SISFE for CNN predictions
    pred_rul_CNN = rul_data_dict['y_pred_CNN']
    sisfe_mean_CNN, sisfe_per_sample_CNN, late_mask_CNN = sisfe(true_rul, pred_rul_CNN, MUL=MUL,
                                                    k_pos=1.8, k_neg=1.0,
                                                    mid_pos=0.5 * MUL, mid_neg=0.25 * MUL)
    
    # Compute SISFE for ANN predictions
    pred_rul_ANN = rul_data_dict['y_pred_ANN']
    sisfe_mean_ANN, sisfe_per_sample_ANN, late_mask_ANN = sisfe(true_rul, pred_rul_ANN, MUL=MUL,
                                                    k_pos=1.8, k_neg=1.0,
                                                    mid_pos=0.5 * MUL, mid_neg=0.25 * MUL)
    
    # Compute SISFE for LSTM predictions
    pred_rul_LSTM = rul_data_dict['y_pred_LSTM']
    sisfe_mean_LSTM, sisfe_per_sample_LSTM, late_mask_LSTM = sisfe(true_rul, pred_rul_LSTM, MUL=MUL,
                                                    k_pos=1.8, k_neg=1.0,
                                                    mid_pos=0.5 * MUL, mid_neg=0.25 * MUL)
    

    
    # Plot SISFE per sample (top) and running RMSE (bottom)
    fig, ax = plt.subplots(figsize=(10, 5))

    idx = np.arange(len(true_rul))

    # Top: SISFE per sample, color-coded for early (blue) vs late (red)
    ax.scatter(idx[late_mask_CNN], sisfe_per_sample_CNN[late_mask_CNN], marker='o', label='Late predictions (over)', zorder=3)
    ax.scatter(idx[~late_mask_CNN], sisfe_per_sample_CNN[~late_mask_CNN], marker='x', label='Early predictions (under)', zorder=3)
    ax.plot(idx, sisfe_per_sample_CNN, linestyle='--', alpha=0.6)
    ax.set_ylabel('SISFE (0..1)')
    ax.set_title('Per-sample SISFE (bounded asymmetric timeliness score)')
    ax.legend()
    ax.grid(True)

     # Print the quantitative result
    print(f"\n--- SISFE Performance Result ---")
    print(f"CNN RUL Predictions SISFE: {sisfe_mean_CNN:.4f}")
    print(f"ANN RUL Predictions SISFE: {0.00:.2f}")
    print(f"LSTM RUL Predictions SISFE: {0.00:.2f}")

    # Store SISFE results in a dictionary
    sisfe_dict['CNN'] = {
        'mean_SISFE': sisfe_mean_CNN,
        'per_sample_SISFE': sisfe_per_sample_CNN,
        'late_mask': late_mask_CNN
    }

    sisfe_dict['ANN'] = {
        'mean_SISFE': sisfe_mean_ANN,
        'per_sample_SISFE': sisfe_per_sample_ANN,
        'late_mask': late_mask_ANN
    }

    sisfe_dict['LSTM'] = {
        'mean_SISFE': sisfe_mean_LSTM,
        'per_sample_SISFE': sisfe_per_sample_LSTM,
        'late_mask': late_mask_LSTM
    }

    return sisfe_dict


"""
Calculates the Prognostics Horizon (PH)

Prognostic Horizon is the difference between the current time index i and EOP utilizing data
accumulated up to the time index i, provided the prediction meets desired specifications
"""
def prognostics_horizon(errors, actual_ruls, units, bound_late, bound_early):
    horizon_units, horizon_percentage_units = [], []
    for unit in np.unique(units):
        ruls_unit = actual_ruls[units == unit]
        ttf = ruls_unit[0]
        errors_unit = errors[units == unit]
        found_horizon = False
        for actual_rul, error in zip(ruls_unit, errors_unit):
            if error < 0 and abs(error) <= bound_late: # late prediction
                horizon_units.append(actual_rul)
                horizon_percentage_units.append(actual_rul/ttf)
                found_horizon = True
                break
            elif error >= 0 and abs(error) <= bound_early: # early prediction
                horizon_units.append(actual_rul)
                horizon_percentage_units.append(actual_rul / ttf)
                found_horizon = True
                break
        if not found_horizon:
            horizon_units.append(0)
            horizon_percentage_units.append(0)

    print(len(np.unique(units)), len(horizon_units))
    return horizon_units, horizon_percentage_units

def plot_PH_metric(actual_rul, predicted_rul, dataset, limits = 0.2, rul_data=None):

    # Simulated actual RUL (ground truth) over time
    ####actual_rul = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
    # Simulated predicted RUL at each time step
    ####predicted_rul = np.array([25, 22, 18, 15, 13, 11, 9, 7, 4.2, 2.1]) #5, 3 

    unit_index = np.arange(len(actual_rul)).tolist()
    #unit_index.sort(reverse=True)


    # Define acceptable error bound (e.g., ±20% of actual RUL)
    alpha = limits * (actual_rul.max() - actual_rul.min())  # 20% late
    #alpha_early = 13  # 20% early
    ####epsilon = 0.2
    # lower_bound = actual_rul * (1 - limits)
    # upper_bound = actual_rul * (1 + limits)
    lower_bound = actual_rul - alpha 
    if  (lower_bound < 0).any():  
        lower_bound[lower_bound < 0] = 0
    
    upper_bound = actual_rul + alpha 

    # Find first time prediction enters and stays within bounds
    ph_index_cnn = None
    for i in range(len(actual_rul)):
        if np.all((predicted_rul[i:] >= lower_bound[i:]) & (predicted_rul[i:] <= upper_bound[i:])):
            ph_index_cnn = i
            break
    ph_index_ann = None
    ph_index_ann_true = None
    ann_predicted_rul = rul_data['y_pred_ANN']
    for i in range(len(actual_rul)):
        if np.all((ann_predicted_rul[i:] >= lower_bound[i:]) & (ann_predicted_rul[i:] <= upper_bound[i:])):
            ph_index_ann = i
            ph_index_ann_true = i
            break
    
    ph_index_lstm = None
    ph_index_lstm_true = None
    lstm_predicted_rul = rul_data['y_pred_LSTM']
    for i in range(len(actual_rul)):
        if np.all((lstm_predicted_rul[i:] >= lower_bound[i:]) & (lstm_predicted_rul[i:] <= upper_bound[i:])):
            ph_index_lstm = i
            ph_index_lstm_true = i
            break
    
    
    # Check if ph_index overlaps were found. Just to separate the vertical lines in the plot.
    if ph_index_ann is not None and ph_index_cnn is not None:
        if ph_index_ann == ph_index_cnn:
            ph_index_ann = ph_index_ann - 0.3  # separate PH for ANN

    if ph_index_lstm is not None and ph_index_cnn is not None:
        if (ph_index_lstm == ph_index_cnn) or (ph_index_lstm == ph_index_ann):
            ph_index_lstm = ph_index_lstm - 0.3  # separate PH for LSTM


    # Compute PH
    if ph_index_cnn is not None:
        prognostic_horizon_cnn = len(actual_rul) - ph_index_cnn
    else:
        prognostic_horizon_cnn = 0  # Never entered acceptable bounds

    if ph_index_ann is not None:
        prognostic_horizon_ann = len(actual_rul) - ph_index_ann
    else:
        prognostic_horizon_ann = 0  # Never entered acceptable bounds

    if ph_index_lstm is not None:
        prognostic_horizon_lstm = len(actual_rul) - ph_index_lstm
    else:
        prognostic_horizon_lstm = 0  # Never entered acceptable bounds


    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(unit_index, actual_rul,  label='Actual RUL', color='blue') # marker='o'
    plt.plot(unit_index, predicted_rul,  label='Predicted RUL - CNN',  color='red', linestyle='--') #marker='x',
    if rul_data is not None:
        plt.plot(unit_index, rul_data['y_pred_ANN'],  label='Predicted RUL - ANN', color='g', linestyle='--')# marker='^', 
        plt.plot(unit_index, rul_data['y_pred_LSTM'],  label='Predicted RUL - LSTM', color='m', linestyle='--')# marker='*', 
    #plt.plot(predicted_rul, label='Predicted RUL - CNN', marker='x', color='red')
    plt.fill_between(range(len(actual_rul)), lower_bound, upper_bound, color='gray', alpha=0.3, label=f'Acceptable Bound (α={limits*100:.0f}%)')
    if ph_index_cnn is not None:
        plt.axvline(ph_index_cnn, color='red', linestyle=':', linewidth=1.5, label=f'PH CNN Start (t={ph_index_cnn})')

    if ph_index_ann is not None:
        plt.axvline(ph_index_ann, color='g', linestyle=':', linewidth=1.5, label=f'PH ANN Start (t={ph_index_ann_true})')

    if ph_index_lstm is not None:
        plt.axvline(ph_index_lstm, color='m', linestyle=':', linewidth=1.5, label=f'PH LSTM Start (t={ph_index_lstm_true})')
    #plt.title(f'Prognostic Horizon = {prognostic_horizon} time units')
    plt.xlabel('Time unit (or Cycle)')
    #plt.xticks([0, 50, 100, 150, 200, 248, 250], ['0', '50', '100', '150', '200', 'EoL', ' '])  # Customize x-ticks
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    fig_path = './images/' + dataset + '_PH.png'
    plt.savefig(fig_path)

    # 3. Print the quantitative result
    print(f"\n--- Prognostics Horizon Result ---")
    print(f"Alpha (Acceptance Error): {limits*100:.1f}%")
    print(f"PH CNN: {prognostic_horizon_cnn} units")
    print(f"PH ANN: {prognostic_horizon_ann} units")
    print(f"PH LSTM: {prognostic_horizon_lstm} units")
    print("---------------------------------------")

"""
Calculates the alpha-lambda accuracy performance

Prediction accuracy at specific time instances; e.g., demand accuracy of prediction to be
within a* 1000/0 after fault detection some defined relative distance
A to actual failure. For example, 200/0 accuracy (i.e., a=0.2) halfway to failure after fault detection (i.e.,
λ=0.5).
"""
def calculate_alpha_lambda_cone(y_true, alpha=0.20, lambda_time=20):
    """
    Calculates the upper and lower boundaries of the alpha-lambda performance cone.

    The cone only opens when the true RUL is less than or equal to lambda_time.

    Args:
        y_true (np.array): Array of true RUL values.
        alpha (float): The fractional error tolerance (e.g., 0.20 for 20%).
        lambda_time (int): The RUL threshold (time units) before failure when the cone starts.

    Returns:
        tuple: (alpha_cone_upper, alpha_cone_lower) arrays.
    """
    n_samples = len(y_true)
    alpha_cone_upper = np.zeros(n_samples)
    alpha_cone_lower = np.zeros(n_samples)

    for i in range(n_samples):
        rul_true = y_true[i]

        # The cone only opens if the system is within the lambda time to failure (RUL <= lambda_time)
        if rul_true <= lambda_time and rul_true > 0:
        #if i >= n_samples - lambda_time:
            upper_bound = rul_true * (1 + alpha)
            lower_bound = rul_true * (1 - alpha)
        
        # Outside the cone area, the bounds simply follow the true RUL line
        else:
            upper_bound = rul_true
            lower_bound = rul_true

        alpha_cone_upper[i] = upper_bound
        alpha_cone_lower[i] = lower_bound

    return alpha_cone_upper, alpha_cone_lower

def plot_alpha_lambda_performance(y_true, y_pred, alpha_cone_upper, alpha_cone_lower, alpha, lambda_time, dataset=None, rul_data=None):
    """
    Plots the RUL prediction against the alpha-lambda cone.

    Args:
        y_true (np.array): Array of true RUL values.
        y_pred (np.array): Array of predicted RUL values.
        alpha_cone_upper (np.array): Upper boundary of the alpha-lambda cone.
        alpha_cone_lower (np.array): Lower boundary of the alpha-lambda cone.
        alpha (float): The alpha value (error tolerance).
        lambda_time (int): The lambda time (start of cone).
    """
    
    # 1. Calculate the percentage of predictions within the cone
    
    # We only care about predictions where RUL_true <= lambda_time and RUL_true > 0
    relevant_indices = np.where((y_true <= lambda_time) & (y_true > 0))[0]
    
    if len(relevant_indices) == 0:
        alpha_coverage = 0.0
        print("No data points found within the lambda time window.")
    else:
        # Check which CNN predictions fall within the cone boundaries
        cnn_predictions_in_cone = (y_pred[relevant_indices] <= alpha_cone_upper[relevant_indices]) & \
                              (y_pred[relevant_indices] >= alpha_cone_lower[relevant_indices])
        
        cnn_alpha_coverage = np.sum(cnn_predictions_in_cone) / len(relevant_indices)

        # Check which ANN predictions fall within the cone boundaries
        ann_y_pred = np.array(rul_data['y_pred_ANN'])
        ann_predictions_in_cone = (ann_y_pred[relevant_indices] <= alpha_cone_upper[relevant_indices]) & \
                              (ann_y_pred[relevant_indices] >= alpha_cone_lower[relevant_indices])
        
        ann_alpha_coverage = np.sum(ann_predictions_in_cone) / len(relevant_indices)

        # Check which LSTM predictions fall within the cone boundaries
        lstm_y_pred = np.array(rul_data['y_pred_LSTM'])
        lstm_predictions_in_cone = (lstm_y_pred[relevant_indices] <= alpha_cone_upper[relevant_indices]) & \
                              (lstm_y_pred[relevant_indices] >= alpha_cone_lower[relevant_indices])
        
        lstm_alpha_coverage = np.sum(lstm_predictions_in_cone) / len(relevant_indices)

    
    # 2. Plotting the results
    
    #plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot the cone area
    ax.fill_between(range(len(y_true)), alpha_cone_lower, alpha_cone_upper, 
                     color='gray', alpha=0.4, 
                     label=r'Acceptable $\alpha$-Cone ($\alpha={:d}\%$)'.format(int(alpha*100)))

    # Plot RUL lines
    ax.plot(y_true, label='True RUL', color='blue') # linewidth=2
    ax.plot(y_pred, label='Predicted RUL - CNN', color='red', linestyle='--')# linewidth=1.5 marker='x', 
    if rul_data is not None:
        ax.plot(rul_data['y_pred_ANN'], label='Predicted RUL - ANN', color='g', linestyle='--') #marker='^', 
        ax.plot(rul_data['y_pred_LSTM'], label='Predicted RUL - LSTM', color='m', linestyle='--') # marker='*', 
    
    # Highlight the lambda time region visually
    #lambda_start_index = np.argmin(np.abs(y_true - lambda_time))
    lambda_start_index = len(y_true) - len(np.where(y_true <= lambda_time)[0])
    ax.axvline(x=lambda_start_index, color='k', linestyle=':', 
                label=r'$\lambda$-Time Threshold ($\lambda={:d}$)'.format(lambda_time), linewidth=1.5)

    #plt.title(r'$\alpha$-$\lambda$ Performance Evaluation ($\alpha={:d}\%$, $\lambda={:d}$)'.format(int(alpha*100), lambda_time))
    plt.xlabel('Time unit (or Cycle)')
    #plt.xticks([0, 50, 100, 150, 200, 248, 250], ['0', '50', '100', '150', '200', 'EoL', ' '])  # Customize x-ticks
    ax.set_xlabel('Time unit (or Cycle)')
    ax.set_ylabel('Remaining Useful Life (RUL)')
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(False)

    # Define zoom area
    x1, x2 = 180, 250   # X-range for zoom
    y1, y2 = 0, 50  # Y-range for zoom

    # Create inset axes (positioned inside main plot)
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)

    # Plot the same data in the inset
    #axins.plot(x, y)
    axins.fill_between(range(len(y_true)), alpha_cone_lower, alpha_cone_upper, 
                     color='gray', alpha=0.4, 
                     label=r'Acceptable $\alpha$-Cone ($\alpha={:d}\%$)'.format(int(alpha*100)))
    axins.plot(y_true, label='True RUL', color='blue') # linewidth=2
    axins.plot(y_pred, label='Predicted RUL - CNN', color='red', linestyle='--')# linewidth=1.5 marker='x', 
    if rul_data is not None:
        axins.plot(rul_data['y_pred_ANN'], label='Predicted RUL - ANN', color='g', linestyle='--') #marker='^', 
        axins.plot(rul_data['y_pred_LSTM'], label='Predicted RUL - LSTM', color='m', linestyle='--') # marker='*', 
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True)

    # Remove tick labels for clarity
    axins.set_xticks([])
    axins.set_yticks([])

    # Draw lines connecting inset to zoomed area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()


    plt.show()
    fig_path = './images/' + dataset + '_alpha-lambda.png'
    plt.savefig(fig_path)
    
    
    # 3. Print the quantitative result
    print(f"\n--- Alpha-Lambda Performance Result ---")
    print(f"Alpha (Acceptance Error): {alpha*100:.1f}%")
    print(f"Lambda (Start Time for Cone): {lambda_time} units")
    print(f"Percentage of CNN Predictions within Cone (Alpha Coverage): {cnn_alpha_coverage*100:.2f}%")
    print(f"Percentage of ANN Predictions within Cone (Alpha Coverage): {ann_alpha_coverage*100:.2f}%")
    print(f"Percentage of LSTM Predictions within Cone (Alpha Coverage): {lstm_alpha_coverage*100:.2f}%")
    print("---------------------------------------")

# Prognostic Horizon (PH) and Alpha-Lambda (α-λ) metrics
def plotPH(actual_ruls, predicted_ruls, dataset=None, rul_data=None): 
    alpha = 0.10  # 5% 10% 20% error tolerance
    plot_PH_metric(actual_ruls, predicted_ruls, dataset, alpha, rul_data)
    #plot_PH_metric(actual_ruls, predicted_ruls, dataset, alpha, rul_data)
    

def plotAlphaLambda(actual_ruls, predicted_ruls, rul_data, dataset=None):
    # 1. Define sample time
    n_steps = len(actual_ruls)

    # True RUL: linearly decreasing from 99 to 0
    y_true = actual_ruls

    # Predicted RUL: starts with high error and converges towards the true RUL near EOL
    y_pred = predicted_ruls
    

    # 2. Set Alpha and Lambda Parameters
    ALPHA = 0.20  # 20% 15% 10% error tolerance
    LAMBDA = 30   # Cone starts when RUL is 30 units or less - 10, 20, 30

    # 3. Calculate Cone Boundaries
    cone_upper, cone_lower = calculate_alpha_lambda_cone(actual_ruls, alpha=ALPHA, lambda_time=LAMBDA)

    # 4. Plot Performance
    plot_alpha_lambda_performance(actual_ruls, predicted_ruls, cone_upper, cone_lower, alpha=ALPHA, lambda_time=LAMBDA, dataset=dataset, rul_data=rul_data)


def calculate_s_score(true_rul, predicted_rul, dict_ANN=None, dict_LSTM=None):
    """
    Calculates the S-score (Symmetric Error Metric) for RUL prediction.

    The S-score is defined as:
    s = sum(s_i) / N
    where s_i is the score for a single prediction:
    s_i = exp(-1/a * e_i) - 1  if e_i < 0 (Early Prediction, penalty 'a')
    s_i = exp(1/b * e_i) - 1   if e_i >= 0 (Late Prediction, penalty 'b')
    e_i = true_rul_i - predicted_rul_i
    """
    N = len(true_rul)
    
    # Define penalty factors (often a=10, b=13 in standard applications)
    # 'a' penalizes late predictions (predicted_rul > true_rul)
    # 'b' penalizes early predictions (predicted_rul < true_rul)
    a = 10
    b = 13
    
    # Calculate the error: e_i = True RUL - Predicted RUL
    error = predicted_rul - true_rul
    
    # Initialize the individual scores array
    s_i = np.zeros_like(error, dtype=float)
    
    # Identify early predictions (Predicted RUL > True RUL, so error < 0)
    early_idx = error < 0
    s_i[early_idx] = np.exp(-1/b * error[early_idx]) - 1
    error_early_pred = error[early_idx]
    
    # Identify late predictions (Predicted RUL <= True RUL, so error >= 0)
    late_idx = error >= 0
    s_i[late_idx] = np.exp(1/a * error[late_idx]) - 1
    error_late_pred = error[late_idx]
    
    # The final S-score is the mean of the individual scores
    S_score = np.mean(s_i) * len(s_i)
    
    return S_score, s_i, early_idx, late_idx, error_early_pred, error_late_pred

def plotScoreMetrics(S_score_total, s_i_scores, early_idx, late_idx, error_early, error_late):
    """
    Plots the S-score against prediction errors for RUL predictions.

    Args:
        true_rul (np.array): Array of true RUL values.
        predicted_rul (np.array): Array of predicted RUL values.
    """
    # --- 3. Plotting ---
    plt.figure(figsize=(8, 6))

    # Plot the True RUL as a reference line
    # plt.plot(time_points, true_rul_at_time, 
    #          label='True RUL', color='k', linestyle='--', linewidth=2)

    # Plot the Predicted RUL, color-coding based on the error type

    # Plot Early Predictions (Predicted RUL > True RUL, which is heavily penalized)
    # plt.scatter(time_points[early_idx], predicted_rul_at_time[early_idx], 
    #             color='r', label='Early Prediction (Predicted > True)', s=50, marker='x')
    # plt.scatter(error_early, s_i_scores[early_idx], 
    #             color='r', label='Early Prediction (Predicted > True)', s=50, marker='x')
    plt.plot(error_early, s_i_scores[early_idx], 
                color='r', label='Early Prediction (Predicted > True)')

    # Plot Late Predictions (Predicted RUL <= True RUL, which is less penalized)
    # plt.scatter(time_points[late_idx], predicted_rul_at_time[late_idx], 
    #             color='b', label='Late Prediction (Predicted $\leq$ True)', s=50, marker='o', alpha=0.6)
    plt.scatter(error_late, s_i_scores[late_idx], 
                color='b', label='Late Prediction (Predicted $\leq$ True)', s=50, marker='o', alpha=0.6)

    # Add a diagonal line for perfect prediction (Predicted RUL = True RUL)
    # This is mainly for visualization when comparing True vs. Predicted RUL values, 
    # but here it's more illustrative of the error zones.
    # plt.plot(time_points, true_rul_at_time, 
    #          color='gray', linestyle=':', alpha=0.7)


    # --- 4. Plot Customization and Labels ---
    plt.title(f'RUL Prognostic Performance (Total S-score: {S_score_total:.2f})', 
            fontsize=16)
    plt.xlabel('Prediction Error', fontsize=14)
    plt.ylabel('S-Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)

    # # Annotate the different scoring zones
    # max_rul = max(np.max(true_rul_at_time), np.max(predicted_rul_at_time))

    # # Early Prediction Zone (where Predicted RUL > True RUL)
    # plt.text(5, max_rul * 0.95, r'Early Prediction (High Penalty: $e^{-a \cdot e}$)', 
    #          color='red', fontsize=10, weight='bold', ha='left')

    # # Late Prediction Zone (where Predicted RUL <= True RUL)
    # plt.text(95, max_rul * 0.95, r'Late Prediction (Lower Penalty: $e^{b \cdot e}$)', 
    #          color='blue', fontsize=10, weight='bold', ha='right')

    plt.tight_layout()
    plt.show()

def plotDefaultScoreMetrics(model_type, dict_CNN=None, dict_ANN=None, dict_LSTM=None, dataset=None):

    # Plot default RMSE and score functions
    early_Score_list_CNN = dict_CNN['S_scores_time_series'][dict_CNN['early_idx']]
    late_Score_list_CNN = dict_CNN['S_scores_time_series'][dict_CNN['late_idx']]

    ##print('dict CNN : ', dict_CNN)

    # S-score and RMSE for CNN
    if model_type == 'CNN' and dict_CNN is not None:
        early_Score_list_CNN.sort()
        late_Score_list_CNN.sort()
        early_Pred_Error_CNN = np.array(dict_CNN['Delta_early'])
        late_Pred_Error_CNN = np.array(dict_CNN['Delta_late'])
        #early_Pred_Error.sort(reverse=True)
        early_Pred_Error_CNN[::-1].sort()
        late_Pred_Error_CNN.sort()

        # RMSE plotting data
        # Plot early and late RMSE list - LSTM
        CNN_squared_error = (dict_CNN['y_true'] - dict_CNN['y_pred'])**2
        delta_early_CNN = np.array(dict_CNN['Delta_early'])
        delta_late_CNN = np.array(dict_CNN['Delta_late'])
        rmse_E_list_CNN = np.sqrt(CNN_squared_error[dict_CNN['early_idx']])
        rmse_L_list_CNN = np.sqrt(CNN_squared_error[dict_CNN['late_idx']])

    # S-score and RMSE for ANN
    if model_type == 'ANN' and dict_ANN is not None:
        # S-score plotting data
        early_Score_list_ANN = dict_ANN['S_scores_time_series'][dict_ANN['early_idx']]
        late_Score_list_ANN = dict_ANN['S_scores_time_series'][dict_ANN['late_idx']]
        early_Score_list_ANN.sort()
        late_Score_list_ANN.sort()
        ##print('ANN early Score list length: ', len(early_Score_list_ANN))
        ##print('ANN late Score list length: ', late_Score_list_ANN)
        early_Pred_Error_ANN = np.array(dict_ANN['Delta_early'])
        late_Pred_Error_ANN = np.array(dict_ANN['Delta_late'])
        ##print('ANN early Pred Error length: ', len(early_Pred_Error_ANN))
        ##print('ANN late Pred Error length: ', late_Pred_Error_ANN)
        #early_Pred_Error.sort(reverse=True)
        early_Pred_Error_ANN[::-1].sort()
        late_Pred_Error_ANN.sort()

        # RMSE plotting data
        # Plot early and late RMSE list - LSTM
        ANN_squared_error = (dict_ANN['y_true'] - dict_ANN['y_pred'])**2
        delta_early_ANN = np.array(dict_ANN['Delta_early'])
        delta_late_ANN = np.array(dict_ANN['Delta_late'])
        #print('LSTM L_rmse_delta: ', len(dict_ANN['Delta_late']))
        #rmse_E_list = [np.sqrt(np.mean((E_rmse_delta[i]-0)**2)) for i in range(len(E_rmse_delta))]
        rmse_E_list_ANN = np.sqrt(ANN_squared_error[dict_ANN['early_idx']])
        rmse_L_list_ANN = np.sqrt(ANN_squared_error[dict_ANN['late_idx']])
    
    # S-score for LSTM
    if model_type == 'LSTM' and dict_LSTM is not None:
        early_Score_list_LSTM = dict_LSTM['S_scores_time_series'][dict_LSTM['early_idx']]
        late_Score_list_LSTM = dict_LSTM['S_scores_time_series'][dict_LSTM['late_idx']]
        early_Score_list_LSTM.sort()
        late_Score_list_LSTM.sort()
        early_Pred_Error_LSTM = np.array(dict_LSTM['Delta_early'])
        late_Pred_Error_LSTM = np.array(dict_LSTM['Delta_late'])
        #early_Pred_Error.sort(reverse=True)
        early_Pred_Error_LSTM[::-1].sort()
        late_Pred_Error_LSTM.sort()

        # RMSE plotting data
        # Plot early and late RMSE list - LSTM
        LSTM_squared_error = (dict_LSTM['y_true'] - dict_LSTM['y_pred'])**2
        delta_early_LSTM = np.array(dict_LSTM['Delta_early'])
        delta_late_LSTM = np.array(dict_LSTM['Delta_late'])
        #####print('LSTM L_rmse_delta: ', len(dict_LSTM['Delta_late']))
        #rmse_E_list = [np.sqrt(np.mean((E_rmse_delta[i]-0)**2)) for i in range(len(E_rmse_delta))]
        rmse_E_list_LSTM = np.sqrt(LSTM_squared_error[dict_LSTM['early_idx']])
        rmse_L_list_LSTM = np.sqrt(LSTM_squared_error[dict_LSTM['late_idx']])
    
    # Plot early and late score list
    T_fig1, T_axes1 = plt.subplots(figsize=(5, 4), sharey=True)
    
    if model_type == 'CNN' and dict_CNN is not None:
        print('Plotting CNN Default Score Metrics - 2 ...')
        plt.plot(early_Pred_Error_CNN, early_Score_list_CNN, color='red', linestyle='-', lw=2)
        plt.plot(late_Pred_Error_CNN, late_Score_list_CNN, color='red', linestyle='-', lw=2, label='S-Score CNN')
        plt.plot(delta_early_CNN, rmse_E_list_CNN, color='k', linestyle='-', lw=2)
        plt.plot(delta_late_CNN, rmse_L_list_CNN, color='k', linestyle='-', lw=2, label='RMSE CNN')

    if model_type == 'ANN' and dict_ANN is not None:
        plt.plot(early_Pred_Error_ANN, early_Score_list_ANN, color='g', linestyle='-', lw=2)
        plt.plot(late_Pred_Error_ANN, late_Score_list_ANN, color='g', linestyle='-', lw=2, label='S-Score ANN')
        plt.plot(delta_early_ANN, rmse_E_list_ANN, color='k', linestyle='-', lw=2)
        plt.plot(delta_late_ANN, rmse_L_list_ANN, color='k', linestyle='-', lw=2, label='RMSE ANN')
    if model_type == 'LSTM' and dict_LSTM is not None:
        plt.plot(early_Pred_Error_LSTM, early_Score_list_LSTM, color='m', linestyle='-', lw=2)
        plt.plot(late_Pred_Error_LSTM, late_Score_list_LSTM, color='m', linestyle='-', lw=2, label='S-Score LSTM')
        plt.plot(delta_early_LSTM, rmse_E_list_LSTM, color='k', linestyle='-', lw=2)
        plt.plot(delta_late_LSTM, rmse_L_list_LSTM, color='k', linestyle='-', lw=2, label='RMSE LSTM')
        
    plt.xlabel("Prediction Error ($\Delta_i$)")
    plt.ylabel("Evaluation metric (Score / RMSE)") 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=10)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    #plt.rc('font', size=10)         # controls default text sizes
    plt.rc('legend', fontsize=8)     # legend fontsize
    #plt.xlim([-50, 50])
    
    # Plot early and late rmse list
    # T_fig2, T_axes2 = plt.subplots(figsize=(5, 4), sharey=True)# Original figsize=(6, 4)
    # plt.plot(early_Pred_Error, early_Score_list, color='red',  linestyle='-', lw=1) #marker='s',
    # plt.plot(late_Pred_Error, late_Score_list, color='red',  linestyle='-', lw=1, label='Score') #marker='s',
    # plt.plot(E_rmse_d, rmse_E_list, color='k',  linestyle='-', lw=1) #marker='o',
    # plt.plot(L_rmse_d, rmse_L_list, color='k',  linestyle='-', lw=1, label='RMSE') #marker='o',

    # plt.xlabel("Prediction Error (di)")
    # plt.ylabel("Evaluation metric (Score / RMSE)") 
    # #plt.legend()
    # plt.legend(loc='best', fancybox=True, shadow=True)
    # plt.rc('axes', titlesize=9)     # fontsize of the axes title
    # plt.rc('axes', labelsize=9)     # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=9)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=9)    # fontsize of the tick labels
    # #plt.rc('font', size=10)         # controls default text sizes
    #plt.rc('legend', fontsize=9)     # legend fontsize
    #plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.legend()
    plt.xlim([-60, 60])

    plt.tight_layout()
    plt.show()
    fig_path = './images/' + dataset + '_' + model_type + '_Default_Score.png'
    plt.savefig(fig_path)

    # Compute Confusion Matrix Metrics
    # CNN S-score CM error metrics 
    CNN_error = dict_CNN['y_pred'] - dict_CNN['y_true']
    CNN_TP_predictions = len(CNN_error[(CNN_error >= -13) & (CNN_error < 0)])
    CNN_FP_predictions = len(CNN_error[CNN_error < -13])
    CNN_FN_predictions = len(CNN_error[(CNN_error >= 0) & (CNN_error <= 10)])
    CNN_TN_predictions = len(CNN_error[CNN_error > 10])

    # ANN S-score CM error metrics 
    ANN_error = dict_ANN['y_pred'] - dict_ANN['y_true']
    ANN_TP_predictions = len(ANN_error[(ANN_error >= -13) & (ANN_error < 0)])
    ANN_FP_predictions = len(ANN_error[ANN_error < -13])
    ANN_FN_predictions = len(ANN_error[(ANN_error >= 0) & (ANN_error <= 10)])
    ANN_TN_predictions = len(ANN_error[ANN_error > 10])

    # LSTM S-score CM error metrics 
    LSTM_error = dict_LSTM['y_pred'] - dict_LSTM['y_true']
    LSTM_TP_predictions = len(LSTM_error[(LSTM_error >= -13) & (LSTM_error < 0)])
    LSTM_FP_predictions = len(LSTM_error[LSTM_error < -13])
    LSTM_FN_predictions = len(LSTM_error[(LSTM_error >= 0) & (LSTM_error <= 10)])
    LSTM_TN_predictions = len(LSTM_error[LSTM_error > 10])  

    # Print the quantitative result
    if model_type == 'CNN':
        print(f"\n--- CNN S-score CM error metrics ---")
        print(f"CNN TP predictions: {CNN_TP_predictions:.2f}")
        print(f"CNN FP predictions : {CNN_FP_predictions:.2f}")
        print(f"CNN FN Predictions : {CNN_FN_predictions:.2f}")
        print(f"CNN TN Predictions : {CNN_TN_predictions:.2f}")
        if (CNN_TP_predictions + CNN_FN_predictions) > 0:
            print(f"CNN TPR : {CNN_TP_predictions/(CNN_TP_predictions + CNN_FN_predictions):.2f}")

        if (CNN_FP_predictions + CNN_TN_predictions) > 0:
            print(f"CNN FPR : {CNN_FP_predictions/(CNN_FP_predictions + CNN_TN_predictions):.2f}")
        # print(CNN_error.min(), CNN_error.max())
        # print('CNN early pred error: ', early_Pred_Error_CNN,  'CNN late pred error: ', late_Pred_Error_CNN)
    
    if model_type == 'ANN':
        print(f"\n--- ANN S-score CM error metrics ---")
        print(f"ANN TP predictions: {ANN_TP_predictions:.2f}")
        print(f"ANN FP predictions : {ANN_FP_predictions:.2f}")
        print(f"ANN FN Predictions : {ANN_FN_predictions:.2f}")
        print(f"ANN TN Predictions : {ANN_TN_predictions:.2f}")
        if (ANN_TP_predictions + ANN_FN_predictions) > 0:
            print(f"ANN TPR : {ANN_TP_predictions/(ANN_TP_predictions + ANN_FN_predictions):.2f}")
        if (ANN_FP_predictions + ANN_TN_predictions) > 0:
            print(f"ANN FPR : {ANN_FP_predictions/(ANN_FP_predictions + ANN_TN_predictions):.2f}")

    if model_type == 'LSTM':
        print(f"\n--- LSTM S-score CM error metrics ---")
        print(f"LSTM TP predictions: {LSTM_TP_predictions:.2f}")
        print(f"LSTM FP predictions : {LSTM_FP_predictions:.2f}")
        print(f"LSTM FN Predictions : {LSTM_FN_predictions:.2f}")
        print(f"LSTM TN Predictions : {LSTM_TN_predictions:.2f}")
        if (LSTM_TP_predictions + LSTM_FN_predictions) > 0: 
            print(f"LSTM TPR : {LSTM_TP_predictions/(LSTM_TP_predictions + LSTM_FN_predictions):.2f}")
        if (LSTM_FP_predictions + LSTM_TN_predictions) > 0:
            print(f"LSTM FPR : {LSTM_FP_predictions/(LSTM_FP_predictions + LSTM_TN_predictions):.2f}")
        print("---------------------------------------")


def plotSISFEMetric(sisfe_dict, sisfe_type, dict_CNN=None, dict_ANN=None, dict_LSTM=None, dataset=None):


#         'SISFE': sisfe_score,
#         'SISFE_time_series': sisfe_values,
#         'SISFE_errors': sisfe_error,
#         'SISFE_late_pred': sisfe_late_pred,
#         'SISFE_early_pred': sisfe_early_pred,
 
    if sisfe_type == 'CNN' and dict_CNN is not None:   
        # SISFE for CNN
        late_mask_CNN = sisfe_dict['CNN']['late_mask'] 
        early_sisfe_list = sisfe_dict['CNN']['per_sample_SISFE'][~late_mask_CNN]
        late_sisfe_list = sisfe_dict['CNN']['per_sample_SISFE'][late_mask_CNN]
        early_sisfe_list.sort()
        late_sisfe_list.sort()
        early_Pred_Error = np.array(dict_CNN['Delta_early'])
        late_Pred_Error = np.array(dict_CNN['Delta_late'])
        early_Pred_Error[::-1].sort()
        late_Pred_Error.sort()

    #      sisfe_dict['CNN'] = {
    #     'mean_SISFE': sisfe_mean_CNN,
    #     'per_sample_SISFE': sisfe_per_sample_CNN,
    #     'late_mask': late_mask_CNN
    # }

    
    # SISFE for ANN
    if sisfe_type == 'ANN' and dict_ANN is not None:
        # SISFE plotting data for ANN
        late_mask_ANN = sisfe_dict['ANN']['late_mask'] 
        early_sisfe_list = sisfe_dict['ANN']['per_sample_SISFE'][~late_mask_ANN]
        late_sisfe_list = sisfe_dict['ANN']['per_sample_SISFE'][late_mask_ANN]
        early_sisfe_list.sort()
        late_sisfe_list.sort()
        early_Pred_Error = np.array(dict_ANN['Delta_early'])
        late_Pred_Error = np.array(dict_ANN['Delta_late'])
        #early_Pred_Error.sort(reverse=True)
        early_Pred_Error[::-1].sort()
        late_Pred_Error.sort()
    
    # SISFE for LSTM
    if sisfe_type == 'LSTM' and dict_LSTM is not None:
        # SISFE plotting data for ANN
        late_mask_LSTM = sisfe_dict['LSTM']['late_mask'] 
        early_sisfe_list = sisfe_dict['LSTM']['per_sample_SISFE'][~late_mask_LSTM]
        late_sisfe_list = sisfe_dict['LSTM']['per_sample_SISFE'][late_mask_LSTM]
        early_sisfe_list.sort()
        late_sisfe_list.sort()
        early_Pred_Error = np.array(dict_LSTM['Delta_early'])
        late_Pred_Error = np.array(dict_LSTM['Delta_late'])
        #early_Pred_Error.sort(reverse=True)
        early_Pred_Error[::-1].sort()
        late_Pred_Error.sort()

            
    # Plot early and late SISFE list
    T_fig1, T_axes1 = plt.subplots(figsize=(6, 4), sharey=True)

    if sisfe_type == 'CNN' and dict_CNN is not None:
        plt.plot(early_Pred_Error, early_sisfe_list, color='red', linestyle='-', lw=2)
        plt.plot(late_Pred_Error, late_sisfe_list, color='red', linestyle='-', lw=2, label='SISFE CNN')
    
    if sisfe_type == 'ANN' and dict_ANN is not None:
        plt.plot(early_Pred_Error, early_sisfe_list, color='g', linestyle='-', lw=2)
        plt.plot(late_Pred_Error, late_sisfe_list, color='g', linestyle='-', lw=2, label='SISFE ANN')
        #print('ANN S-score plotting not implemented yet.')
        
    if sisfe_type == 'LSTM' and dict_LSTM is not None:
        plt.plot(early_Pred_Error, early_sisfe_list, color='m', linestyle='-', lw=2)
        plt.plot(late_Pred_Error, late_sisfe_list, color='m', linestyle='-', lw=2, label='SISFE LSTM')
        #print('LSTM S-score plotting not implemented yet.')
        
       
        
    plt.xlabel("Prediction Error ($\Delta_i$)")
    plt.ylabel("Evaluation metric (SISFE)") 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=10)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    #plt.rc('font', size=10)         # controls default text sizes
    plt.rc('legend', fontsize=8)     # legend fontsize  
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.xlim([-40, 40])
    plt.tight_layout()
    plt.show()
    
    fig_path = './images/' + dataset + '_Default_Score.png'
    plt.savefig(fig_path)

#########################################################################################
# Additional Utility Functions for Saving and Loading RUL Data and Metrics

# Save true and predicted RUL values to a text file
def save_RUL_to_file(true_RUL_data, pred_RUL_data, dataset, regressor_type):
    folder_path = './lists/'  # Replace with your desired folder path fig_path = './images/' + dataset + '_RUL.png'

    # Save the predicted and true RUL lists to text files
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    file_name_1 = dataset + regressor_type + '_true_RUL_data.txt'
    file_path_1 = os.path.join(folder_path, file_name_1)
    with open(file_path_1, 'w') as file:
        for item in true_RUL_data:
            file.write(f"{item}\n")

    file_name_2 = dataset + regressor_type + '_pred_RUL_data.txt'
    file_path_2 = os.path.join(folder_path, file_name_2)
    with open(file_path_2, 'w') as file:
        for item in pred_RUL_data:
            file.write(f"{item}\n")

    # # Read the list back
    # with open('true_RUL_data.txt', 'r') as file:
    #     loaded_list = [int(line.strip()) for line in file]
    # #print(pred_RUL_sorted)

# Calculate and save the performance metrics for the test dataset using the true and predicted RUL values.
def calculate_save_all_metrics(true_RUL, pred_RUL, regressor_type, dataset=None, a1=10, a2=13):
    # Example usage with dummy data
    
    # y_true= np.array(true_RUL)
    # y_pred = np.array(pred_RUL) #pred_RUL 
    #y_pred_example = np.maximum(y_pred_example, 0)

    # Score function parameters
    a1 = a1  # Example value for a1
    a2 = a2  # Example value for a2

    # Calculate all metrics
    metrics_data = calculate_all_metrics(true_RUL, pred_RUL, a1, a2, regressor_type)
    #print(f"Calculated Metrics: {metrics_data}")

    rmse = metrics_data['RMSE']
    sf = metrics_data['S_score']

    # Save metrics data to a file using pickle for later analysis or plotting

    # Specify folder and file path
    folder_path = './dicts/'
    file_name = dataset + metrics_data['reg_type'] + '_metrics_data.pkl'
    file_path = os.path.join(folder_path, file_name)

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Serialize
    with open(file_path, "wb") as file:
        pickle.dump(metrics_data, file)


    # Deserialize
    # Load dictionary from a file
    # with open(folder_path, 'rb') as file:
    #     loaded_dict = pickle.load(file)
    # print(loaded_dict)

    return rmse, sf

# Load actual and predicted RUL text files for all regressor models (CNN, ANN, LSTM) and store them 
# in a dictionary for consolidated plotting and analysis.
def load_actual_pred_RUL_from_files(dataset):
    
    # Store Load actual and predicted RUL text files in a dictionary
    RUL_data_dict = {'y_true_CNN': [], 'y_pred_CNN': [],
                    'y_true_ANN': [], 'y_pred_ANN': [],
                    'y_true_LSTM': [], 'y_pred_LSTM': []}

    # Consolidate RUL prediction plot
    # 1. Load actual and predicted RUL text files
    # Use float() when reading because prediction files may contain floating point values.
    # If integer values are required later, cast/round where appropriate.

    # File 1 and 2: True RUL data and Predicted RUL data for CNN regressor
    folder_path = './lists/'
    file_name_1 = dataset + 'CNN' + '_true_RUL_data.txt'
    file_path_1 = os.path.join(folder_path, file_name_1)
    with open(file_path_1, 'r') as file:
        RUL_data_dict['y_true_CNN'] = [float(line.strip()) for line in file]
            
    file_name_2 = dataset + 'CNN' + '_pred_RUL_data.txt'
    file_path_2 = os.path.join(folder_path, file_name_2)
    with open(file_path_2, 'r') as file:
        RUL_data_dict['y_pred_CNN'] = [float(line.strip()) for line in file]
    
    # File 3 and 4: True RUL data and Predicted RUL data for ANN regressor
    file_name_3 = dataset + 'ANN' + '_true_RUL_data.txt'
    file_path_3 = os.path.join(folder_path, file_name_3)
    with open(file_path_3, 'r') as file:
        RUL_data_dict['y_true_ANN'] = [float(line.strip()) for line in file]
            
    file_name_4 = dataset + 'ANN' + '_pred_RUL_data.txt'
    file_path_4 = os.path.join(folder_path, file_name_4)
    with open(file_path_4, 'r') as file:
        RUL_data_dict['y_pred_ANN'] = [float(line.strip()) for line in file]

    # File 5 and 6: True RUL data and Predicted RUL data for LSTM regressor
    file_name_5 = dataset + 'LSTM' + '_true_RUL_data.txt'
    file_path_5 = os.path.join(folder_path, file_name_5)
    with open(file_path_5, 'r') as file:
        RUL_data_dict['y_true_LSTM'] = [float(line.strip()) for line in file]
            
    file_name_6 = dataset + 'LSTM' + '_pred_RUL_data.txt'
    file_path_6 = os.path.join(folder_path, file_name_6)
    with open(file_path_6, 'r') as file:
        RUL_data_dict['y_pred_LSTM'] = [float(line.strip()) for line in file]

    return RUL_data_dict

# Summarize the prognostics performance metrics - RMSE, S-score, SISFE
def load_metrics_data(dataset):
    
    # Load RMSE metrics_data from the saved pickle file

    # Specify folder and file path
    folder_path = './dicts/'

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    file_name_1 = dataset + 'CNN' + '_metrics_data.pkl'
    file_path_1 = os.path.join(folder_path, file_name_1)

    # Deserialize - Load dictionary from a file
    with open(file_path_1, 'rb') as file:
        metrics_dict_CNN = pickle.load(file)
    #print(metrics_dict_1['RMSE'])

    file_name_2 = dataset + 'ANN' + '_metrics_data.pkl'
    file_path_2 = os.path.join(folder_path, file_name_2)

    # Deserialize - Load dictionary from a file
    with open(file_path_2, 'rb') as file:
        metrics_dict_ANN = pickle.load(file)


    file_name_3 = dataset + 'LSTM' + '_metrics_data.pkl'
    file_path_3 = os.path.join(folder_path, file_name_3)

    # Deserialize - Load dictionary from a file
    with open(file_path_3, 'rb') as file:
        metrics_dict_LSTM = pickle.load(file)

    return metrics_dict_CNN, metrics_dict_ANN, metrics_dict_LSTM



