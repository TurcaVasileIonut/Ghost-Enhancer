from torch.utils.data import DataLoader

from Data.CustomDataset import CustomDataset

High_Data = ["../Data/HighToLowDataset/HIGH/celea_60000_SFD", "../Data/HighToLowDataset/HIGH/SRtrainset_2",
             "../Data/HighToLowDataset/HIGH/vggface2/vggcrop_train_lp10"]
Low_Data = ["../Data/HighToLowDataset/LOW/wider_lnew"]


class HighToLowDataLoader:
    def __init__(self):
        self.dataset = CustomDataset(High_Data, Low_Data)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)

    def denormalize(self, tensor):
        return self.dataset.denormalize(tensor)
