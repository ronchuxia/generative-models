from torch.utils.data import Dataset
from sklearn.datasets import make_blobs, make_circles, make_moons
import torch


class MoonsDataset(Dataset):
    def __init__(self, n_samples, noise):
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise

        x, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CirclesDataset(Dataset):
    def __init__(self, n_samples, noise):
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise

        x, y = make_circles(n_samples=n_samples, shuffle=True, noise=noise)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class BlobsDataset(Dataset):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        x, y = make_blobs(n_samples=n_samples, shuffle=True)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MoonsCirclesDataset(Dataset):
    def __init__(self, n_samples, noise_moons, noise_circles):
        super().__init__()
        self.n_samples = n_samples

        x0, y0 = make_moons(n_samples=n_samples, shuffle=True, noise=noise_moons)
        x1, y1 = make_circles(n_samples=n_samples, shuffle=True, noise=noise_circles)
        self.x0 = torch.Tensor(x0)
        self.x1 = torch.Tensor(x1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x0[index], self.x1[index]

