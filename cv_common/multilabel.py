"""fastai loaders for validated absolute image paths and list-valued labels."""
def build_loaders(frame, labels, image_size=224, batch_size=32, augment=True):
    from fastai.vision.all import (DataBlock, ImageBlock, MultiCategoryBlock,
                                  ColReader, ColSplitter, Resize, aug_transforms)
    # Preserve all rows without singleton training batches in the BatchNorm head.
    n_train = int((~frame['is_valid']).sum())
    if batch_size < 2 or n_train < 2:
        raise ValueError('At least two training images and batch_size >= 2 are required.')
    batch_size = min(batch_size, n_train)
    while batch_size > 2 and n_train % batch_size == 1:
        batch_size -= 1
    while n_train % batch_size == 1:
        batch_size += 1
    block = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(vocab=labels)),
        get_x=ColReader('filename'), get_y=ColReader('labels'),
        splitter=ColSplitter('is_valid'), item_tfms=Resize(image_size),
        batch_tfms=aug_transforms(size=image_size) if augment else None)
    return block.dataloaders(frame, bs=batch_size, num_workers=0, drop_last=False)
