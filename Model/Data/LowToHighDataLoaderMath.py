from torch.utils.data import DataLoader

from Data.CustomDataset import CustomDataset
import os

from Data.CustomDatasetMathRes import CustomDatasetMathRes


class LowToHighDataLoaderMath:
    def __init__(self):
        base_path_high = "../Data/LowToHighDataset/HIGH"

        Data = []

        # Populate the High_Data list with paths to all folders in the HIGH directory
        for root, dirs, files in os.walk(base_path_high):
            for name in dirs:
                Data.append(os.path.join(root, name))
        self.dataset = CustomDatasetMathRes(Data)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)

    def denormalize(self, tensor):
        return self.dataset.denormalize(tensor)
