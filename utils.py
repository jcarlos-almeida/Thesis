#################################################################################
# utils.py
# Utility functions for data pre-processing, and learning rate finder.
# The functions are designed to be modular and reusable across different models and datasets.
# This file was adapted from https://github.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/ 
# as part of the paper titled "Remaining Useful Life Estimation with Variational 
# Recurrent Autoencoders: A Comparative Study on the C-MAPSS Dataset" Published under the CCBY-NC-NDlicense 
# (http://creativecommons.org/licenses/bync-nd/4.0/.
# This file was adapted by Jose Carlos Almeida on 07-01-2026 to fix the learning rate finder 
# implementation, add an alternative condition scaler and test data generation function, and other adjustments.
#####################################################################################################################
import math
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import keras
try:
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback
except Exception:
    # Fallback to standalone Keras if tensorflow.keras is not available/resolvable
    from keras import backend as K
    from keras.models import load_model
    from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback


# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

def add_operating_condition(df):
    df_op_cond = df.copy()
    
    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                        df_op_cond['setting_2'].astype(str) + '_' + \
                        df_op_cond['setting_3'].astype(str)
    
    return df_op_cond

def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train, df_test

##################################################################################################
#---------------------------------- ALTERNATIVE CONDITION SCALER --------------------------------
# JC 07-01-2026: alternative condition scaler function
##################################################################################################
def test_condition_scaler(df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    #scaler = MinMaxScaler() 
    for condition in df_test['op_cond'].unique():
        scaler.fit(df_test.loc[df_test['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_test

def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)
    
    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    
    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]
    
    return df

def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]
        
def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    data_gen = (list(gen_train_data(df[df['unit_nr']==unit_nr], sequence_length, columns))
               for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array

def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]  

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    label_gen = [gen_labels(df[df['unit_nr']==unit_nr], sequence_length, label) 
                for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array

def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values
        
    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]  
        
#############################################################################################
#----------------------------------- ALTERNATIVE TEST DATA GENERATION ---------------------
# jc 07-01-2026: alternative test data generation function
###############################################################################################
def gen_test_data_2(df, sequence_length, columns, mask_value):
    #num_elements = df.shape[0]
    
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values
    
    num_elements = data_matrix.shape[0]
           
    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data_matrix[start:stop, :]
# ---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Get start and stop indexexes for each unit in the dataframe
def get_unit_start_stop_indexes(df):
    unit_start_stop_indexes = {}
    for unit_nr in df['unit_nr'].unique():
        unit_data = df[df['unit_nr']==unit_nr]
        start_index = unit_data.index.min()
        stop_index = unit_data.index.max()
        unit_start_stop_indexes[unit_nr] = (start_index, stop_index)
    return unit_start_stop_indexes

# --------------------------------------------------------------------------------------------------- 
# Create sequences for unit data in test set   
def create_sequences_unit_data(df, sequence_length, sensors):  
    # create sequences for specific unit data in test set 
    test_gen = list(gen_test_data_2(df, sequence_length, sensors, -99.)) # test_gen
    x_test = np.concatenate(test_gen).astype(np.float32)
    return x_test
#---------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# Get alternative test data  
def get_test_data_df(dataset, sensors, sequence_length, alpha, threshold):
	# files
	dir_path = './data/'
	#train_file = 'train_'+dataset+'.txt'
	test_file = 'test_'+dataset+'.txt'
    # columns
	index_names = ['unit_nr', 'time_cycles']
	setting_names = ['setting_1', 'setting_2', 'setting_3']
	sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
	col_names = index_names + setting_names + sensor_names
    
    # data readout
	# train = pd.read_csv((dir_path+train_file), sep=r'\s+', header=None, 
	# 				 names=col_names)
	test = pd.read_csv((dir_path+test_file), sep=r'\s+', header=None, 
					 names=col_names)
	y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None, 
					 names=['RemainingUsefulLife'])

    # create RUL values according to the piece-wise target function
	test = add_remaining_useful_life(test)
	test['RUL'].clip(upper=threshold, inplace=True) 
	#y_PW_test = test['RUL'].values

    # remove unused sensors
	drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
	#X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
	X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
	X_test_pre = test_condition_scaler(X_test_pre, sensors)

    # exponential smoothing
	#X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
	X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    # get start and stop indexes for each unit
	#unit_start_stop_idx_dict = get_unit_start_stop_indexes(X_test_pre)

	# create sequences for test 
	# test_gen = (list(gen_test_data_2(X_test_pre[X_test_pre['unit_nr']==unit_nr], sequence_length, sensors, -99.))
	# 		   for unit_nr in X_test_pre['unit_nr'].unique())
	# x_test = np.concatenate(list(test_gen)).astype(np.float32)
	
	return X_test_pre # unit_start_stop_idx_dict
#---------------------------------------------------------------------------------------------------

# --------------------------------------- MAIN GET DATA FUNCTION ---------------------------------------	
def get_data(dataset, sensors, sequence_length, alpha, threshold):
	# files
	dir_path = './data/'
	train_file = 'train_'+dataset+'.txt'
	test_file = 'test_'+dataset+'.txt'
    # columns
	index_names = ['unit_nr', 'time_cycles']
	setting_names = ['setting_1', 'setting_2', 'setting_3']
	sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
	col_names = index_names + setting_names + sensor_names
    # data readout
	train = pd.read_csv((dir_path+train_file), sep=r'\s+', header=None, 
					 names=col_names)
	test = pd.read_csv((dir_path+test_file), sep=r'\s+', header=None, 
					 names=col_names)
	y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None, 
					 names=['RemainingUsefulLife'])

    # create RUL values according to the piece-wise target function
	train = add_remaining_useful_life(train)
	train['RUL'].clip(upper=threshold, inplace=True)

    # remove unused sensors
	drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
	X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
	X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
	X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # exponential smoothing
	X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
	X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

	# train-val split
	gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
	# generate the train/val for *each* sample -> for that we iterate over the train and val units we want
	# this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
	# i.e. train_unit and val_unit are not a single value but a set of training/vali units
	for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()): 
		train_unit = X_train_pre['unit_nr'].unique()[train_unit]  # gss returns indexes and index starts at 1
		val_unit = X_train_pre['unit_nr'].unique()[val_unit]

		x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
		y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)
		
		x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
		y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

	# create sequences for test 
	test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr']==unit_nr], sequence_length, sensors, -99.))
			   for unit_nr in X_test_pre['unit_nr'].unique())
	x_test = np.concatenate(list(test_gen)).astype(np.float32)
	
	return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']
# ---------------------------------------------------------------------------------------------------


# ----------------------------------------- FIND OPTIMAL LR  ----------------------------------------
class LRFinder:
    """
    Cyclical LR, code tailored from:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1, **kw_fit):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=[callback],
                       **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(generator)
            except (ValueError, NotImplementedError) as e:
                raise e('`steps_per_epoch=None` is only valid for a'
                        ' generator based on the '
                        '`keras.utils.Sequence`'
                        ' class. Please specify `steps_per_epoch` '
                        'or use the `keras.utils.Sequence` class.')
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs * steps_per_epoch))

        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch,
                                                      logs: self.on_batch_end(batch, logs))

        self.model.fit_generator(generator=generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[callback],
                                 **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]
# ---------------------------------------------------------------------------------------------------


#----------------------------------- Updated LearningRateFinder Class --------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LearningRateFinder(tf.keras.callbacks.Callback):
    def __init__(self, target_model, smoothing_factor=0.98): 
        super().__init__()
        self.smoothing_factor = smoothing_factor
        self._target_model = target_model 
        
        self.lrs = []
        self.losses = []
        self.best_loss = 1e9
        self.avg_loss = 0.0
        self.current_step = 0
        self.total_steps = 0 
        self._start_lr = 0.0  # To store the initial learning rate 
        self._cloned_model = None  # To hold the cloned model

    # Updated LearningRateFinder.find() method (replacing the old version)

    def find(self, x_train, y_train, start_lr=1e-6, end_lr=1e-1, batch_size=32, epochs=5, initial_weights=None, **kwargs):
        print("Starting Learning Rate Finder...")
        
        # Pre-check original model
        if self._target_model.optimizer is None:
            raise ValueError("The model must be compiled with an optimizer before running the LR finder.")

        # 1. Clone Model
        self._cloned_model = tf.keras.models.clone_model(self._target_model)
        
        # Get original optimizer config and loss
        # original_optimizer_config = self._target_model.optimizer.get_config()
        # original_optimizer = self._target_model.optimizer.__class__.from_config(original_optimizer_config)
        # original_loss = self._target_model.loss

        # # 2. Compile Cloned Model
        # self._cloned_model.compile(
        #     optimizer=original_optimizer,
        #     loss=original_loss
        # )

        # 3. Setup steps and multiplier
        data_size = len(x_train)
        steps_per_epoch = int(np.ceil(data_size / batch_size))
        self.total_steps = steps_per_epoch * epochs
        self.lr_multiplier = (end_lr / start_lr) ** (1.0 / self.total_steps)
        
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=data_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 🌟 FINAL WORKAROUND: Avoid tf.keras.backend.set_value 🌟
        # Instead, modify the optimizer config directly, or use the setter if it exists.
        # We will explicitly create a new optimizer with the starting LR and re-compile.

        # Get original loss (assuming this is safe)
        original_loss = self._target_model.loss

        # 1. Create a NEW OPTIMIZER with the correct starting LR
        NewOptimizerClass = self._target_model.optimizer.__class__
        new_optimizer = NewOptimizerClass(learning_rate=start_lr)

        # 2. Re-compile the cloned model with the new optimizer
        self._cloned_model.compile(
            optimizer=new_optimizer,
            loss=original_loss
        )

        # 🌟 CRITICAL FIX: Set LR directly on the cloned model's optimizer object 🌟
        # This avoids the string/object conflict experienced in on_train_begin.
        #tf.keras.backend.set_value(self._cloned_model.optimizer.learning_rate, start_lr)
        
        # 4. Transfer weights
        if initial_weights:
            self._cloned_model.load_weights(initial_weights)
        else:
            self._cloned_model.set_weights(self._target_model.get_weights())
        
        # 5. Run training
        # Keras will set self.model = self._cloned_model here.
        self._cloned_model.fit( 
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[self],
            verbose=0,
            **kwargs
        )

        print(f"Finder complete. Total steps: {self.current_step}")
    
    # ... (on_train_begin method remains the same if defined) ...
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        
        # --- 1. EMA Smoothing and Loss Tracking ---
        if self.current_step == 0:
            self.avg_loss = loss
        else:
            self.avg_loss = (self.smoothing_factor * self.avg_loss) + \
                            ((1.0 - self.smoothing_factor) * loss)
        
        # Get the current LR value (reading is usually safe)
        # Use self.model which Keras sets to self._cloned_model during fit()
        current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        self.lrs.append(current_lr)
        
        # Apply bias correction
        bias_corrected_loss = self.avg_loss / (1.0 - self.smoothing_factor**(self.current_step + 1))
        self.losses.append(bias_corrected_loss)

        # Update best loss for stopping condition
        if self.current_step > 1 and self.avg_loss < self.best_loss:
            self.best_loss = self.avg_loss
        
        # --- 2. Update Learning Rate ---
        new_lr = current_lr * self.lr_multiplier
        
        # 🌟 FINAL FIX: Use the native TensorFlow Variable .assign() method 🌟
        # This directly updates the learning rate Tensor, bypassing serialization conflicts.
        self.model.optimizer.learning_rate.assign(new_lr)
        
        # --- 3. Stopping Condition ---
        if self.current_step > 1 and self.avg_loss > (4 * self.best_loss):
            print("\nLoss exploded. Stopping search.")
            self.model.stop_training = True
            
        self.current_step += 1

        # Note: If you keep using `self.model` in on_train_batch_end, it should 
        # theoretically work, but using the explicit `self._cloned_model` is safer 
        # here to avoid any final, subtle internal Keras confusion.

    
    # ⚠️ REMOVE OR COMMENT OUT THIS METHOD TO PREVENT THE ERROR ⚠️
    # def on_train_begin(self, logs=None):
    #     # This logic is now handled correctly in find()
    #     pass
        # ... (The plot method remains the same) ...

    def plot(self):
        # ... (implementation as previously provided) ...
        skip_start = 20
        skip_end = 5
        
        lrs = self.lrs[skip_start:-skip_end]
        losses = self.losses[skip_start:-skip_end]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Smoothed Loss (Log Scale)")
        plt.title("Learning Rate Finder: Loss vs. Learning Rate")
        plt.grid(True, which="both", ls="--")
        plt.show()
        
        min_loss_idx = np.argmin(losses)
        suggested_lr_index = np.argmin(np.gradient(losses)) 
        suggested_lr = lrs[suggested_lr_index]
        print(f"\n💡 Suggested optimal LR (Based on steepest slope): {suggested_lr:.2e}")
        print(f"   Consider a range around {suggested_lr:.2e} to {lrs[min_loss_idx]:.2e}")

# 🛠️ FIX: Learning Rate Finder Implementation
def start_lr_finder(model, x_train, y_train, batch_size):

    #######################################################################################################################
    # 1. Instantiate and Compile Model (Use a dummy high LR, it will be reset by the finder)
    # NOTE: compile WITHOUT passing a string loss when using a custom Model that implements train_step / test_step.
    # Passing a string loss can trigger internal Keras logic that expects certain attributes (leading to
    # AttributeError: 'str' object has no attribute 'name'). The custom training loop computes losses internally.
    init_lr = 1e-7
    #opt = keras.optimizers.Adam(learning_rate=init_lr)
    opt=tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Ensure optimizer has an 'lr' attribute as a tf.Variable and update optimizer hyper
    opt.lr = tf.Variable(init_lr, dtype=tf.float32)
    try:
        # update optimizer internals to reference this variable (best-effort)
        opt._set_hyper('learning_rate', opt.lr)
    except Exception:
        # If internal API differs, we still have opt.lr available for legacy utilities
        pass

    model.compile(optimizer=opt, loss='mse')


    # 4. Instantiate and Run Finder
    # We pass the data and all search parameters here. Use the notebook batch_size variable if desired.
    lr_finder = LearningRateFinder(model)
    lr_finder.find(
        x_train=x_train,
        y_train=y_train,
        start_lr=1e-6, # Starting learning rate
        end_lr=1e-1,   # Ending learning rate
        batch_size=batch_size,
        epochs=5
    )

    # 5. Plot Results
    lr_finder.plot()






# --------------------------------------------- RESULTS  --------------------------------------------
def get_model(path):
    saved_VRAE_model = load_model(path, compile=False)
    
    # return encoder, regressor
    return saved_VRAE_model.layers[1], saved_VRAE_model.layers[2]

def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

def score(y_true, y_hat):
  res = 0
  for true, hat in zip(y_true, y_hat):
    subs = hat - true
    if subs < 0:
      res = res + np.exp(-subs/10)[0]-1
    else:
      res = res + np.exp(subs/13)[0]-1
  print("score: ", res)


# ---------------------------------------------------------------------------------------------------