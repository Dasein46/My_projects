"""fastai loaders for validated absolute image paths and list-valued labels."""
def build_loaders(frame, labels, image_size=224, batch_size=32, augment=True):
    from fastai.vision.all import (DataBlock, ImageBlock, MultiCategoryBlock,
                                  ColReader, ColSplitter, Resize, aug_transforms)
    block = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(vocab=labels)),
        get_x=ColReader('filename'), get_y=ColReader('labels'),
        splitter=ColSplitter('is_valid'), item_tfms=Resize(image_size),
        batch_tfms=aug_transforms(size=image_size) if augment else None)
    return block.dataloaders(frame, bs=batch_size, num_workers=0)
