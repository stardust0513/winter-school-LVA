import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, sig, spectrum=False):
        sig = torch.tensor(sig, dtype=torch.float32)
        # normalization
        sig = sig - torch.mean(sig, dim=-1, keepdim=True)
        sig = sig / torch.std(sig, dim=-1, keepdim=True)
        if spectrum == False:
            self.sig = sig
        else:
            # pwelch spectrum
            # fold signal into segments and average the FFT results
            segment_length = 2048
            overlap = 0.5
            stride = int(segment_length * (1 - overlap))
            # (N, C, num_segments, seglen)
            segments = sig.unfold(dimension=2, size=segment_length, step=stride)
            # -> (N, num_segments, C, seglen)
            segments = segments.permute(0, 2, 1, 3)

            fft = torch.fft.rfft(segments, dim=-1, norm = "ortho")
            power = fft.abs()
            self.sig = power.mean(dim=1)  # (N, C, F)

    def __len__(self):
        return self.sig.shape[0]

    def __getitem__(self, idx):
        return self.sig[idx]

class IMS():
    def __init__(
        self,
        data_path,
        test_idx=1,
        sensor_idx=1
    ):
        self.data_path = os.path.join(data_path, f"test{test_idx}")
        assert test_idx in [1,2,3,4], "test_idx must be in [1,2,3,4]"
        assert sensor_idx > 0, "sensor_idx must be positive integer"
        self.test_idx = test_idx
        self.sensor_idx = sensor_idx - 1

    def read(self, sample_num):
        assert sample_num > 0, "sample_num must be positive integer"
        
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith(".txt")]
        file = files[sample_num-1]
        df = pd.read_csv(os.path.join(self.data_path, file), sep="\t", header=None)
        sig = df[self.sensor_idx].values[np.newaxis,:]

        return sig

    def read_list(self, sample_list):
        
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith(".txt")]
        sig_train = []
        for num in sample_list:
            file = files[num]
            df = pd.read_csv(os.path.join(self.data_path, file), sep="\t", header=None)
            sig_train.append(df[self.sensor_idx].values[np.newaxis,:])
        
        # sample_list里没有的是sig_test
        sig_test = []
        for num in range(len(files)):
            if num not in sample_list:
                file = files[num]
                df = pd.read_csv(os.path.join(self.data_path, file), sep="\t", header=None)
                sig_test.append(df[self.sensor_idx].values[np.newaxis,:])
        
        return np.array(sig_train), np.array(sig_test)
    
    def rms_trend(self):
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith(".txt")]

        rms_list = []
        for num in range(len(files)):
            file = files[num]
            df = pd.read_csv(os.path.join(self.data_path, file), sep="\t", header=None)
            sig = df[self.sensor_idx].values
            rms = np.sqrt(np.mean(sig**2))
            rms_list.append(rms)
        
        return np.array(rms_list)
    
    def unfold(self, sig, segment_length, overlap):
        """
        Unfold a signal into overlapping segments.
        
        Parameters:
        sig : ndarray of shape (N, 1, L)
        segment_length : int
        overlap : float between 0 and 1
        
        Returns:
        ndarray of shape (N * num_segments, 1, segment_length)
        """
        N, C, L = sig.shape
        assert C == 1, "Second dimension must be 1"

        stride = int(segment_length * (1 - overlap))
        if stride <= 0:
            raise ValueError("Overlap too high, resulting in non-positive stride")

        num_segments = (L - segment_length) // stride + 1
        segments = []

        for i in range(num_segments):
            start = i * stride
            end = start + segment_length
            segment = sig[:, :, start:end]  # shape (N, 1, segment_length)
            segments.append(segment)

        segments = np.stack(segments, axis=1)  # shape (N, num_segments, 1, segment_length)
        return segments.reshape(-1, 1, segment_length)
    
    def __call__(
        self,
        train_samples,
        **kwargs,
    ):
        segment_length = kwargs.get("segment_length", 20480)
        overlap = kwargs.get("overlap", 0)
        assert 0 <= overlap < 1, "overlap must be in [0, 1)"
        
        # update the data_path based on the test_idx

        sample_list = [i for i in range(train_samples)]
        sig_train, sig_test = self.read_list(sample_list)
        sig_train = self.unfold(sig_train, segment_length, overlap)
        sig_test = self.unfold(sig_test, segment_length, overlap)

        dataset_train = CustomDataset(
            sig = sig_train,
            spectrum=kwargs.get("spectrum", False),
        )
        dataset_test = CustomDataset(
            sig = sig_test,
            spectrum=kwargs.get("spectrum", False),
        )

        return dataset_train, dataset_test