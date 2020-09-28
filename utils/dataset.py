from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize

LABEL_TO_IDX = {
    'staphylococcus_epidermidis': 0,
    'klebsiella_pneumoniae': 1,
    'staphylococcus_aureus': 2,
    'moraxella_catarrhalis': 3,
    'c_kefir': 4,
    'ent_cloacae': 5
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BacteriaDataset(Dataset):
    def __init__(self, images, labels=None, masks=None, transform=None):
        self.images = images
        self.labels = labels
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.masks is not None:
            mask = self.masks[idx]

        if self.transform is not None:
            if self.masks is not None:
                image, mask = self.transform(
                    image=image,
                    segmentation_maps=mask
                )
            else:
                image = self.transform(image=image)

        image = to_tensor(image)
        image = normalize(image, MEAN, STD)

        sample = {'image': image}

        if self.labels is not None:
            sample['label'] = self.labels[idx]

        if self.masks is not None:
            mask = to_tensor(mask)
            sample['mask'] = mask

        return sample
