import h5py
import os, fnmatch
import numpy as np   
from scipy.interpolate import interp1d
from tensorflow.keras.utils import Sequence

class LoadNR: 

    '''
    LoadNR is developed to load SXS¹ data. The metadata and waveforms (wfs) files must have the 
    format ID_metadata.txt and ID_WFfile.h5, where "ID" is the identification simulation number 
    and "WFfile" is the wf file name. In particular, we adopt "ID_rhOverM.h5" and "ID_rMPsi4.h5" 
    names in our database. The list of ID numbers can be accessed by the attribut "IDs".

    The class initializes with two parameters:
    · path_load: the path directory where the database is stored;
    · wf_file: the waveform name.
    

    There are four methods:
    → masses(): returns the (m1, m2) masses of the binaries after the simulation relaxation 
      time;

    → spins(): returns the (χ⃗1, χ⃗2) dimensionless spins of the binaries after the simulation 
      relaxation time;

    → eccentricity(): returns the orbital eccentricity of the binaries after the simulation 
      relaxation time;

    → waveform(modes): returns (t, wfs) with the respective array shapes, (number-simulations, ) 
      and (number-simulations, modes-list-size). The "modes" list must have tuples of "l" and 
      "m" numbers. See example below.


    - E.g.:
      nr = loadNR(path_name="database/", wf_file="rhOverM")

      print(nr.IDs)

      m1, m2 = nr.masses()

      t, h = nr.waveform([(2,2), (2,1), (3,2), (3,3)])


    ¹ https://data.black-holes.org/waveforms/catalog.html
    '''
    
    def __init__(self, path_load="database/", wf_file="rhOverM"):
        
        # reading metadata files:
        metadata, IDs = [], []
        for file in os.listdir(path_load):
            if fnmatch.fnmatch(file, "*metadata.txt"):
                metadata.append(path_load + file)
                IDs.append(file.split("_")[0])
        
        self.IDs = IDs
        self.metadata = metadata
        self.wf_files = [f"{path_load}{ID}_{wf_file}.h5" for ID in IDs] 
        
    
    def masses(self):

        m1, m2 = [], []
        for data in self.metadata:
            with open(data) as File:
                
                flag = True
                for line in File:
                    if 'relaxed-mass1' in line or 'reference-mass1' in line:
                        m1.append(float(line.split("=")[1]))

                    elif 'relaxed-mass2' in line or 'reference-mass2' in line:
                        m2.append(float(line.split("=")[1]))
                        flag = False
                        break
            
                if flag:
                    raise TypeError(f"\n*** WARNING: reference masses were not found! File:{data} ***\n")

        return np.asarray(m1), np.asarray(m2)
    
    
    def spins(self): 

        chi1, chi2 = [], []
        for data in self.metadata:
            with open(data) as File:

                flag = True
                for line in File:
                    if 'relaxed-dimensionless-spin1' in line or 'reference-dimensionless-spin1' in line:
                        chi1.append([float(x) for x in line.split("=")[1].split(",")])
                        
                    elif 'relaxed-dimensionless-spin2' in line or 'reference-dimensionless-spin2' in line:
                        chi2.append([float(x) for x in line.split("=")[1].split(",")])
                        flag = False
                        break

                if flag:
                    raise TypeError(f"\n*** WARNING: reference spins were not found! File:{data} ***\n")

        return np.asarray(chi1), np.asarray(chi2)


    def eccentricity(self):
        
        e = []
        for data in self.metadata:
            with open(data) as File:

                flag = True
                for line in File:
                    if 'relaxed-eccentricity =' in line or 'reference-eccentricity =' in line:
                        try: e.append(float(line.split("=")[1].split("<")[-1]))
                        except: e.append(np.nan)
                        flag = False
                        break
                
                if flag:
                    raise TypeError(f"\n*** WARNING: reference eccentricity was not found! File:{data} ***\n")

        return np.asarray(e)

    
    def __relaxed_time(self): 
        
        t0 = []
        for data in self.metadata:
            with open(data) as File:

                flag = True
                for line in File:
                    if 'relaxed-measurement-time =' in line or 'reference-time =' in line:
                        t0.append(float(line.split("=")[1]))
                        flag = False
                        break

                if flag:
                    raise TypeError(f"\n*** WARNING: relaxed time was not found! File:{data} ***\n")

        return np.asarray(t0)

    
    def waveform(self, modes):

        t_relax = self.__relaxed_time()
        
        t, wf, shift = [], [], []
        for t0, file in zip(t_relax, self.wf_files):    
            try: extract = h5py.File(file, 'r')['OutermostExtraction.dir']
            except: extract = h5py.File(file, 'r')['Extrapolated_N3.dir']
                
            time = extract['Y_l2_m2.dat'][()].T[0]
            wf22 = extract['Y_l2_m2.dat'][()].T[1] + 1j*extract['Y_l2_m2.dat'][()].T[2]

            j = np.argmax(time >= t0)
            time, wf22 = time[j:], wf22[j:]

            shift.append(time[np.argmax(abs(wf22))])
            time -= shift[-1]
            t.append(time)

            wflm = [wf22 if l==m==2
                    else extract[f'Y_l{l}_m{m}.dat'][()].T[1][j:] 
                    + 1j*extract[f'Y_l{l}_m{m}.dat'][()].T[2][j:]
                    for l, m in modes]
                    
            wf.append(wflm)

        t = np.asarray(t, dtype="object")
        self.shift = np.asarray(shift, dtype="object")

        if len(modes) == 1: wf = np.asarray(wf, dtype="object").T[0]
        else: wf = np.asarray(wf, dtype="object")
            
        return t, wf
    
    
class DsetBuilder(LoadNR):

    '''
    DsetBuilder uses loadNR to access the binaries parameters and their waveforms (wfs). 
    The wfs are interpolated and reformed to start at the shortest database initial time 
    and finish after 100 solar masses of the dominant mode amplitude peak. The time 
    interpolation provides more data points during merger-ringdown stages of the 
    coalescence.


    The class initializes with five parameters:
    · path_load: the path directory where the NR database is stored;
    · wf_file: the waveform name;
    · path_save: the path/name of the built dataset;
    · Nt: the timeseries size integer number;
    · modes: a list of tuples containing the "l" and "m" mode numbers.


    It returns in a single h5 file:
    > dataset "X": the binaries parameters (mass-ratio, spins, and eccentricity) 
      linearized for the range [0,1];

    > dataset "y": the real and imaginary parts of the waveforms;

    > dataset "wf_IDs": the list of the identification simulation numbers;

    - attribute "modes": the list of modes used;

    - attribute "range_coeff": the coefficient numbers of each parameter linear 
      transformation;
    
    - attribute "shift": the shift float number computed in all waveforms;

    - attribute "norm": the dataset normalization float number;

    - attribute "time": the nonlinear time array.
    '''

    def __init__(self, path_load="database/", wf_file="rhOverM", path_save="dataset/dset_wfs", Nt=2048, modes=[(2,2)]):
        super().__init__(path_load, wf_file)

        self.Nt = Nt
        self.modes = modes
        self.path_save = path_save


    def __call__(self):

        m1, m2 = self.masses()
        chi1, chi2 = self.spins()
        e = self.eccentricity()
        q = m1/m2

        N_wfs = q.size
        Nf = N_wfs*len(self.modes)
        l_max = self.modes[-1][0]

        time, wfs = self.waveform(self.modes)

        t0 = max([t[0] for t in time])
        t = self._nonlinearTime(np.linspace(t0, 100, self.Nt))

        xl = np.zeros((Nf, l_max-1)) # 2 <= l <= 4
        xm = np.zeros((Nf, l_max-2)) # l-1 <= m <= l

        x1 = np.empty(Nf) # q
        x2, x3, x4 = np.empty(Nf), np.empty(Nf), np.empty(Nf) # chi11, chi12, chi13
        x5, x6, x7 = np.empty(Nf), np.empty(Nf), np.empty(Nf) # chi21, chi22, chi23
        x8 = np.empty(Nf) # e

        wf_IDs = np.empty(Nf)
        y = np.empty((Nf, self.Nt, 2))
        peak = []

        j = 0
        for i, (ID, wf) in enumerate(zip(self.IDs, wfs)):
            for (l, m), hlm in zip(self.modes, wf):

                xl[j][l-2] = 1.0
                xm[j][l-m] = 1.0

                x1[j] = q[i]
                x2[j], x3[j], x4[j] = chi1[i][0], chi1[i][1], chi1[i][2]
                x5[j], x6[j], x7[j] = chi2[i][0], chi2[i][1], chi2[i][2]
                x8[j] = e[i]

                h = interp1d(time[i], hlm, kind='nearest', fill_value='extrapolate')(t)

                peak.append(max(abs(h)))

                if i == 0: shift = np.unwrap(np.angle(h))[np.argmax(abs(h))]

                h *= np.exp(-1j*shift)

                y[j] = np.transpose([h.real, h.imag])
                wf_IDs[j] = ID
                
                j += 1

        # normalization over the dataset:
        norm = max(peak)
        y /= norm
        
        x1, c11, c12 = self.__newRange(x1)
        
        x2, c21, c22 = self.__newRange(x2)
        x3, c31, c32 = self.__newRange(x3)
        x4, c41, c42 = self.__newRange(x4)
        
        x5, c51, c52 = self.__newRange(x5)
        x6, c61, c62 = self.__newRange(x6)
        x7, c71, c72 = self.__newRange(x7)
        
        x8, c81, c82 = self.__newRange(x8)

        Cij = [[c11, c12],
               [c21, c22],
               [c31, c32],
               [c41, c42],
               [c51, c52],
               [c61, c62],
               [c71, c72],
               [c81, c82]]

        X = np.concatenate([xl, xm, x1, x2, x3, x4, x5, x6, x7, x8], axis=1)

        with h5py.File(self.path_save + ".h5", "w") as dset:
            dset.create_dataset("X", data=X.astype("float32"))
            dset.create_dataset("y", data=y.astype("float32"))
            dset.create_dataset("wf_IDs", data=wf_IDs.astype("int32"))
            dset.attrs['range_coeff'] = Cij
            dset.attrs['modes'] = self.modes
            dset.attrs['shift'] = shift
            dset.attrs['norm'] = norm
            dset.attrs['time'] = t

            
    def _nonlinearTime(self, u, a=0.0005, b=500):

        g = lambda x: np.tanh(a*(x - u[0] + b))

        c1 = (u[-1] - u[0])/(g(u[-1]) - g(u[0]))
        c2 = u[0] - c1*g(u[0])

        y = c1*g(u) + c2

        return y
    

    def __newRange(self, x, ya=0.0, yb=1.0):

        xa, xb = np.nanmin(x), np.nanmax(x)

        c1 = (yb - ya)/(xb - xa)
        c2 = ya - xa*c1

        x = np.expand_dims(c1*x + c2, axis=1)

        return x, c1, c2

    
    def paramBack(self, y, Cij):
        return np.asarray([(j-c2)/c1 for j, (c1, c2) in zip(y, Cij)])



class DataGen(Sequence):
        
    '''
    DataGen is a batch dataset generator for TensorFlow/Keras framework.


    The class initializes with four parameters:
    · IDs: the index dataset list;
    · batch_size: the batch-size integer number;
    · shuffle: the boolean parameter for shuffling the IDs numbers;
    · path_dset: the dataset path/name to be loaded.


    The method attached_data() returns the dataset attributes. 
    See DsetBuilder documentation.


    It returns the (X, y) batch, where:
    > X is the binaries parameters;
    > y is the real and imaginary parts of the waveforms.
    '''
    
    def __init__(self, IDs, batch_size, shuffle, path_dset):
    
        self.IDs = IDs    
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path_dset = path_dset
        self.on_epoch_end()
        
        
    def __len__(self):
                
        return int(np.floor(len(self.IDs)/self.batch_size))

    
    def __add__(self, other):
        total_IDs = np.concatenate([self.IDs, other.IDs])
        
        return DataGen(total_IDs, self.batch_size, self.shuffle, self.path_dset)
        
    
    def __getitem__(self, index):
            
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batchIDs = [self.IDs[i] for i in indexes]

        # Generate data
        return self.__data_generation(batchIDs)

    
    def on_epoch_end(self):
                    
        self.indexes = np.arange(len(self.IDs))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
    def attached_data(self):
        
        with h5py.File(self.path_dset + ".h5", "r") as f:
            wf_IDs = np.array(f['wf_IDs'])
            Cij = f.attrs['range_coeff']
            modes = f.attrs['modes']
            shift = f.attrs['shift']
            norm = f.attrs['norm']
            t = f.attrs['time']
        
        wfIDs = []
        for index in range(self.__len__()):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            batchIDs = [self.IDs[i] for i in indexes]
            wfIDs.append([wf_IDs[i] for i in batchIDs])
                
        return wfIDs, Cij, modes, shift, norm, t
        
            
    def __data_generation(self, batchIDs):
                      
        with h5py.File(self.path_dset + ".h5", "r") as f:
            X = np.array([f['X'][ID] for ID in batchIDs])
            y = np.array([f['y'][ID] for ID in batchIDs])
        
        return X, y
    
