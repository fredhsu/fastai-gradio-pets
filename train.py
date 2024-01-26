from fastai.vision.all import *
from fastai.data.all import untar_data, URLs


path = untar_data(URLs.PETS) / "images"

# Download the images using fastai ImageDataLoaders
dls = ImageDataLoaders.from_name_re(
    path, get_image_files(path), pat="(.+)_\d+.jpg", item_tfms=Resize(224)
)
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
learn.export()
