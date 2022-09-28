import h5py
import os, time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .wfdset import DsetBuilder, DataGen
from .evaltoolkit import Mismatch
        
class UNet(keras.Model):
    
    '''    
    The UNet is a class that receives real and imaginary waveform parts as CNN channels and 
    predicts the same timeseries pair.

    - input-shape:  (None, waveform-length, 2)
    - output-shape: (None, waveform-length, 2)

    To setup the UNet, it is necessary to inform:
    · filters: the list of CNN number of filters by layer;
    · kernel_size: the CNN kernel-size value;
    · Nf: the dataset size;
    · Nt: the waveform length;
    · shownet: the boolean flag for showing the network layer shapes.

    - E.g.: 
        keys = {'filters':[32, 64, 128, 256, 512],
                'kernel_size':3,
                'Nf':8046,
                'Nt':2048,
                'shownet':True}
    '''
    
    def __init__(self, keys, **kwargs):
        super(UNet, self).__init__(keys, **kwargs)
        
        self.keys = keys
        self.shownet = keys['shownet']
        n = len(keys['filters'])
        
        self.encoder_conv = [layers.Conv1D(f, keys['kernel_size'], activation='relu', padding='same', name=f"encoder_conv_{f}") 
                             for f in keys['filters'][:-1]] 

        self.pool = layers.MaxPooling1D(2)
        
        self.middle = layers.Conv1D(keys['filters'][-1], keys['kernel_size'], padding="same", name=f"middle_conv_{keys['filters'][-1]}")

        self.up = layers.UpSampling1D(2)
        
        self.decoder_conv1 = [layers.Conv1D(f, keys['kernel_size'], activation='relu', padding='same', name=f"decoder_conv1_{f}") 
                              for f in keys['filters'][n-2::-1]] 
        
        self.decoder_conv2 = [layers.Conv1D(f, keys['kernel_size'], activation='relu', padding='same', name=f"decoder_conv2_{f}") 
                              for f in keys['filters'][n-2::-1]] 
                
        self.concat = layers.Concatenate()
        
        self.out = layers.Conv1D(2, 1, activation='tanh', padding='same', name=f"output_conv")        

    
    @tf.function
    def call(self, x):
        
        if self.shownet: tf.print(f"\nIn:{x.shape}\n")
                
        convs = []
        for conv in self.encoder_conv:
            x = conv(x)
            
            if self.shownet: tf.print(f"x:{x.shape}")
            convs.append(x)
                
            x = self.pool(x)
        
        x = self.middle(x)
        if self.shownet: tf.print(f"x:{x.shape}")
        
        for conv1, conv2, z in zip(self.decoder_conv1, self.decoder_conv2, convs[::-1]):
            x = self.up(x)
            x = conv1(x)
            x = self.concat([x, z])
            x = conv2(x)

            if self.shownet: tf.print(f"x:{x.shape}")

        x = self.out(x)

        if self.shownet: tf.print(f"\nOut:{x.shape}\n")

        return x
        
    
    def get_config(self):
        return self.keys

    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
class FitUNet(UNet):
    
    '''
    The FitUNet is a child class of UNet. It receives the setup UNet dictionary with: 
    - filters: the list of CNN number of filters by layer;
    - kernel_size: the CNN kernel-size value;
    - Nf: the dataset size;
    - Nt: the waveform length;
    - shownet: the boolean flag for showing the network layer shapes.

    * E.g.: 
        keys = {'filters':[32, 64, 128, 256, 512],
                'kernel_size':3,
                'Nf':8046,
                'Nt':2048,
                'shownet':True}
    
    
    This class initializes the mean squared error loss function and the Adagrad 
    optimizer for learning-rate = 0.001. 
    
    
    There are four methods:
    → train(): it trains the model according to the training and validation data.
      Check the function parameters in "train.__doc__".
    
    → kfold(): it uses the train() method to provide a K-fold validation.
      Check the function parameters in "kfold.__doc__".
    
    → plot_metrics(save, save_path): it plots the training and validation losses
      evolution for train() or kfold(). 
      · save: boolean key to allow saving.
      · save_path: path where the figure is saved.
      
    → plot_histogram(save, save_path): it plots the mismatch histograms of training
      and testing data. The mismach is evaluated with the inputs data and its prediction. 
      · save: boolean key to allow saving.
      · save_path: path where the figure is saved.
      
    → plot_waveforms(save, save_path): it takes three pairs of training and testing 
      data at random to be plotted for comparison.
      · save: boolean key to allow saving.
      · save_path: path where the figure is saved.
    '''
    
    def __init__(self, keys, **kwargs):
        super().__init__(keys, **kwargs)
    
        self.lossfun = keras.losses.mean_squared_error
        
        self.train_loss = keras.metrics.Mean(name='train_loss')
        
        self.val_loss = keras.metrics.Mean(name='val_loss')
                        
        self.opt = keras.optimizers.Adagrad(learning_rate=0.001)

    @property
    def metrics(self):
        return [self.train_loss]
    
    
    @tf.function
    def train_step(self, nr_wf):
        
        with tf.GradientTape() as tape:
            nn_wf = self(nr_wf)
            
            loss = tf.reduce_mean(self.lossfun(nr_wf, nn_wf), axis=0, keepdims=True)
                        
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        
        self.train_loss.update_state(loss)
        
        return {m.name: m.result() for m in self.metrics}
        
    
    @tf.function
    def test_step(self, nr_wf):
        
        nn_wf = self(nr_wf)

        loss = tf.reduce_mean(self.lossfun(nr_wf, nn_wf), axis=0, keepdims=True)
            
        self.val_loss.update_state(loss)
        
    
    def train(self, path_dset, train_IDs=None, val_IDs=None, test_IDs=None, epochs=1, batch_size=1, shuffle=True, 
              verbose=True, patience=0, min_delta=1e-8, save=False, path_unet='./unet', _klabel=''):
                 
        '''
        Input arguments: 
        
        · path_dset: dataset path/name;
        · train_IDs: list of indexes corresponding to the dataset training part;
        · val_IDs: list of indexes corresponding to the dataset validation part;
        · test_IDs: list of indexes corresponding to the dataset testing part;
        · epochs: number of epochs for training;
        · batch_size: data batch-size number;
        . shuffle: boolean flag to allow shuffling data in the DataGen object;
        . verbose: boolean flag to allow printing the metric status;
        · patience: integer value of how many epochs to wait for better training results
          before stopping training;
        . min_delta: the value tolerated above the best training loss before stopping 
          training (default:1e-8);
        · save: boolean key to allow saving;
        · path_unet: the path/name given to the trained u-net.
        '''
        
        gen_args = {'batch_size':batch_size, 'shuffle':shuffle, 'path_dset':path_dset}
        
        self.path_dset = path_dset
        self.path_unet = path_unet
        self.model_name = path_unet.split("/")[-1]
                       
        if train_IDs is None:
            Nf = self.get_config()['Nf']
            N_test = int(0.1*Nf)
            N_train = Nf - N_test

            IDs = np.random.permutation(Nf)
            self.test_IDs = IDs[:N_test]
            self.train_IDs = IDs[N_test:]
            self.val_IDs = val_IDs

            train_gen = DataGen(self.train_IDs, **gen_args)

        else:     
            self.train_IDs = train_IDs
            self.val_IDs = val_IDs
            self.test_IDs = test_IDs

            N_train, N_val = train_IDs.size, val_IDs.size

            try:
                train_gen = DataGen(self.train_IDs, **gen_args)
                val_gen = DataGen(self.val_IDs, **gen_args)

            except:
                raise TypeError("For validation tasks, please, inform training, validation, and testing IDs.") 
                
        best, wait = 1.0, 0        
        wf_train, wf_val = [], []
            
        for epoch in range(1, epochs+1):

            start_time = time.time()
            if verbose: print(f"#{_klabel}Epoch {epoch}/{epochs}:")

            for step, batch in enumerate(train_gen, start=1):
                metrics = self.train_step(batch[1])
                
                if verbose and (step % 10 == 0):
                    print(f"Batch:{int(100*step/N_train)}%", end=' - ')
                    print(f"waveform-metric:{metrics['train_loss']:.4e}", end='\r', flush=True)
                
            if verbose: print()

            train_loss = self.train_loss.result().numpy()
            self.train_loss.reset_states()
            wf_train.append(train_loss)

            if val_IDs is not None:
                for batch in val_gen:
                    self.test_step(batch[1])

                val_loss = self.val_loss.result().numpy()
                self.val_loss.reset_states()
                wf_val.append(val_loss)

                if verbose:
                    print(f"Validation:  waveform-metric:{val_loss:.4e}")
                    print(f"Time taken: {time.time() - start_time:.2f}s\n")

                delta = abs(train_loss - val_loss)

                if patience > 0: wait += 1
                    
                if delta < best:
                    best = delta
                    wait = 0

                elif delta < (best + min_delta):
                    wait = 0

                if wait > patience:
                    break     

        if val_IDs is None: self.val_IDs, metrics = [0], [wf_train]
        else: metrics = [wf_train, wf_val]

        self.recorded_metrics = [metrics]

        if save:
            self.save(path_unet)
            
            if _klabel == '':
                with h5py.File(path_unet + "_info.h5", "w") as info:
                    info.create_dataset("metrics", data=metrics)
                    info.attrs["train_IDs"] = self.train_IDs
                    info.attrs["val_IDs"] = self.val_IDs
                    info.attrs["test_IDs"] = self.test_IDs
    
    
    def kfold(self, path_dset, kfold=3, epochs=1, batch_size=1, shuffle=True, verbose=True, patience=10, min_delta=1e-8, 
              save=False, path_unet='./test'):
            
        '''
        Input arguments: 
        
        · path_dset: dataset path/name;
        · kfold: the number of folds for validation;
        · epochs: number of epochs for training;
        · batch_size: data batch-size number;
        . shuffle: boolean flag to allow shuffling data in the DataGen object;
        . verbose: boolean flag to allow printing the metric status;
        · patience: integer value of how many epochs to wait for better training results
          before stopping training;
        . min_delta: the value tolerated above the best training loss before stopping 
          training (default:1e-8);
        · save: boolean key to allow saving;
        · path_unet: the path/name given to the trained u-net.
        '''    
        
        args = {'path_dset':path_dset, 
                'epochs':epochs, 
                'batch_size':batch_size, 
                'shuffle':shuffle, 
                'verbose':verbose, 
                'patience':patience, 
                'min_delta':min_delta,
                'save':save,
                'path_unet':path_unet}
        
        Nf = self.get_config()['Nf']
        IDs = np.random.permutation(Nf)

        N_test = int(0.1*Nf)
        test_IDs = IDs[:N_test]
        IDs = IDs[N_test:]

        N_val = (Nf - N_test)//kfold
        N_train = (Nf - N_test) - N_val
        k_metrics = []
        
        print(f"Training:{N_train} - Validation:{N_val} - Testing:{N_test}\n")        
        
        for k in range(kfold):
            val_IDs = IDs[k*N_val:(k+1)*N_val]     
            train_IDs = np.concatenate([IDs[:k*N_val], IDs[(k+1)*N_val:]], axis=0)

            self.train(train_IDs=train_IDs, val_IDs=val_IDs, test_IDs=test_IDs, _klabel=f'{k+1}-Fold, ', **args)
                
            k_metrics.append(self.recorded_metrics[0])
                
        self.recorded_metrics = k_metrics
        self.train_IDs = train_IDs
        self.val_IDs = val_IDs
        self.test_IDs = test_IDs
        
        if save:
            with h5py.File(path_unet + "_info.h5", "w") as info:
                for i, metrics in enumerate(k_metrics, start=1):
                    info.create_dataset(f"{i}-metrics", data=metrics)

                info.attrs["train_IDs"] = self.train_IDs
                info.attrs["val_IDs"] = self.val_IDs
                info.attrs["test_IDs"] = self.test_IDs

                
    def plot_metrics(self, save=False, save_path='./'):

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        K = len(self.recorded_metrics)

        plt.figure(figsize=(10, 6))

        for k, k_metrics in enumerate(self.recorded_metrics, start=1):
            if K > 1: klabel = f"{k}-Fold: "
            else: klabel = ""

            for i, metric in enumerate(k_metrics):

                if i == 0: 
                    Set = "train"
                    marker = '-'
                else: 
                    Set = "val"
                    marker = '--'

                x = range(1, 1+len(metric))

                plt.plot(x, metric, marker, label=klabel+f"waveform-{Set}")

                plt.ylabel("Loss", fontsize=12)
                plt.xlabel("Epochs", fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True)

        if save: plt.savefig(f"{save_path}{self.model_name}_metrics.png")
        else: plt.show()

            
    def plot_histogram(self, save=False, save_path='./'):
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        args = {'batch_size':100, 'shuffle':True, 'path_dset':self.path_dset}

        if len(self.val_IDs) == 1: train_IDs = self.train_IDs
        else: train_IDs = np.concatenate([self.train_IDs, self.val_IDs])

        train_gen = DataGen(train_IDs, **args)
        test_gen = DataGen(self.test_IDs, **args)

        wf_IDs, Cij, modes, shift, norm, t = train_gen.attached_data()

        l_size = modes[-1][0] - 1
        m_size = modes[-1][0] - modes[-1][-1] + 1

        dset = DsetBuilder()

        for label, data_gen in zip(["Train", "Test"], [train_gen, test_gen]):

            wf_loss = []
            for nr_p, nr_wf in data_gen:

                nn_wf = self(nr_wf).numpy()

                for wf1, wf2 in zip(nr_wf, nn_wf):

                    h1 = (wf1.T[0] + 1j*wf1.T[1])
                    h2 = (wf2.T[0] + 1j*wf2.T[1])

                    wf_loss.append(Mismatch(t, h1, h2))  

            wf_loss = np.asarray(wf_loss)

            print(f"[{label}] wf mismatch: max={wf_loss.max()}, min={wf_loss.min()}, mean={wf_loss.mean()}")

            plt.figure(figsize=(10,6))    
            plt.title(f"{label}: Waveform", fontsize=14)
            plt.hist(wf_loss, bins=wf_loss.size, histtype="step", linewidth=2)
            plt.ylabel("Number of waveforms", fontsize=14)
            plt.xlabel("Mismatch", fontsize=14)
            if save: plt.savefig(f"{save_path}{self.model_name}_{label.lower()}_histogram.png")
            else: plt.show()


    def plot_waveforms(self, save=False, save_path='./'):
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        args = {'batch_size':1, 'shuffle':True, 'path_dset':self.path_dset}

        if len(self.val_IDs) == 1: train_IDs = self.train_IDs
        else: train_IDs = np.concatenate([self.train_IDs, self.val_IDs])

        j = np.random.randint(low=0, high=train_IDs.size, size=3)
        k = np.random.randint(low=0, high=self.test_IDs.size, size=3)

        train_gen = DataGen(train_IDs[j], **args)
        test_gen = DataGen(self.test_IDs[k], **args)

        wf_IDs, Cij, modes, shift, norm, t = train_gen.attached_data()

        l_size = modes[-1][0] - 1
        m_size = modes[-1][0] - modes[-1][-1] + 1

        dset = DsetBuilder()

        for label, data_gen in zip(["Train", "Test"], [train_gen, test_gen]):

            fig, ax = plt.subplots(3, 2, figsize=(24,14))

            for i, (X, y) in enumerate(data_gen):

                y_pred = self(y).numpy()

                p = dset.paramBack(X[0][l_size+m_size:], Cij)
                p = np.round(p, 4)

                print(f"[{label}] NR values: q = {p[0]:.2}", end=", ") 
                print(f"chi1 = ({p[1]:.2e}, {p[2]:.2e}, {p[3]:.2e})", end=", ") 
                print(f"chi2 = ({p[4]:.2e}, {p[5]:.2e}, {p[6]:.2e})", end=", ") 
                print(f"e = {p[7]:.1e}")

                l = int(np.argmax(X[0][0:l_size])+2)
                m = l - int(np.argmax(X[0][l_size:l_size+m_size])) 

                h = norm*(y[0].T[0] + 1j*y[0].T[1])*np.exp(1j*shift)
                h_pred = norm*(y_pred[0].T[0] + 1j*y_pred[0].T[1])*np.exp(1j*shift)

                ax[i, 0].plot(t, h.real)
                ax[i, 0].plot(t, h_pred.real, '--')
                ax[i, 0].grid(True)

                ax[i, 1].plot(t, h.imag, label=f'NR: l={l}, m={m}')
                ax[i, 1].plot(t, h_pred.imag, '--', label=f'NN: l={l}, m={m}')
                ax[i, 1].set_xlim(-100, 100)
                ax[i, 1].legend()
                ax[i, 1].grid(True)

            if save: plt.savefig(f"{save_path}{self.model_name}_{label.lower()}_waveforms.png")
            else: plt.show()
            