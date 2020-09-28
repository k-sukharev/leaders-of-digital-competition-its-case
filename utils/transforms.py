import imgaug.augmenters as iaa

from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def train_transform(image, segmentation_maps=None):
    image_aug = iaa.Sequential([
        iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))
    ], random_order=False)

    geom_aug = iaa.Sequential([
        iaa.flip.Fliplr(p=0.5),
        iaa.flip.Flipud(p=0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
            rotate=(-180, 180),
            shear=(20, 20),
            mode='reflect'
        ),
    ], random_order=False)

    geom_aug_deterministic = geom_aug.to_deterministic()
    image = geom_aug_deterministic.augment(image=image)
    image = image_aug(image=image)

    if segmentation_maps is None:
        return image

    segmentation_maps = geom_aug_deterministic.augment(image=segmentation_maps)
    return image, segmentation_maps


def valid_transform(image, segmentation_maps=None):
    if segmentation_maps is None:
        return image

    segmentation_maps = SegmentationMapsOnImage(
        segmentation_maps,
        shape=image.shape
    )
    return image, segmentation_maps.arr
