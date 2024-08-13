from torch.utils.data import DataLoader

from Data.CustomDataset import CustomDataset
import os

class LowToHighDataLoader:
    def __init__(self):
        base_path_high = "../Data/LowToHighDataset/HIGH"
        base_path_low = "../Data/LowToHighDataset/LOW"

        High_Data = []
        Low_Data = []

        # Populate the High_Data list with paths to all folders in the HIGH directory
        for root, dirs, files in os.walk(base_path_high):
            for name in dirs:
                High_Data.append(os.path.join(root, name))

        # Populate the Low_Data list with paths to all folders in the LOW directory
        for root, dirs, files in os.walk(base_path_low):
            for name in dirs:
                Low_Data.append(os.path.join(root, name))
        self.dataset = CustomDataset(Low_Data, High_Data)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)

    def denormalize(self, tensor):
        return self.dataset.denormalize(tensor)
