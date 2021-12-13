# Self-supervised denoising for massive noisy images

----

## Usage:

Update the first few lines in the main script, before running it.

```python
image_path = '/home/feng/workspace/self2noise/s9.tif' # 1000x512x512 experimental dataset
training_index = 's9' # the identity for the training, can be a random string
model_directory = f'/home/feng/workspace/self2noise/model_noisy_to_clean_model_{training_index}' # saving the model
threshold = 1 # bottleneck threshold
traing_images = 900 # images used to train the model, the more the better, but the training will be slow
dim = 512 # cut the input images to small pieces for a larger batch size, as normalization layers prefer large batch size
batch_size = 16 # traing batch size, need adjusting according to the GPU memory, 16-32 is good enough
gpu_id = 0 # the GPU to use
zoom_factor = 1 # zoom images before denoising, giving better performance when the shared structure are small in pixels
enhance_contrast = False # enhance contrast
n_loops = 200 # training loops
check_intervals = 4 # the training intervals to see the training result
test_image_index = 0 # the image used to visually evaluate the model's performance in the real time
lr = 0.001 # learing rate
loss = 'mae' # mae performs better than mse when signal is sparse
private_key = None # your private key for Telegram bot. If set to 'None', you will not receive training messages from the bot.
private_id = None # your private id for Telegram bot. If set to 'None', you will not receive training memssages.
```

## Links to the Dataset:

- [Platinum clusters](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.201911068)
- [Platinum nano particles](https://github.com/fengwang/Noise2Atom/releases/download/experimental/2019_03_26-15.35abcdf.2048px.8.8pm.tif.too_large_0_100.tif)
- [Platinum Atomic Images](https://doi.org/10.1186/s42649-020-00041-8)
- [DR9](https://www.legacysurvey.org/dr9/description/)
- [WQY](http://wenq.org/en/)
- [Tribolium nuclei](https://csbdeep.bioimagecomputing.com/scenarios/denoisingtribolium/)


## Pre-trained models

We released two pre-trained models with a GUI for STEM images, for win64 platform. You can find it from the release.


## License

BSD-3









