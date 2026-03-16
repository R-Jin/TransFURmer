# TransFURmer


## Using container

### Build container
Build the container using the provided `minerva_env.def` file with this command:

```bash
apptainer build env.sif minerva_env.def
```

This will create the image `env.sif`

### Running image
Here is how you run commands within the container:

```bash
apptainer exec --nv env.sif <cmd>
```

Use `--nv` to allow container to use host's drivers (Needed for using CUDA with torch).

Example running `main.py`:
```bash
apptainer exec --nv env.sif python main.py
```

### Adding or removing dependencies 
Add or remove dependencies from `environment.yml` and then [build container](#build-container)

## Create environment
```bash
conda env create -f environment-local.yml
```

# Note
Might not be good to downsize whole image to 128x128 because fur strands details will be non existent making it look blurry. This might confuse model and undo high fidelity fur texture.

However we could zoom in on it by cropping first to a smaller size and then upsize the image to 128x128. This could lead to finer detail on fur. 

```py
class RandomScaleCrop:
    def __init__(self, output_size, scale_range=(0.75, 1.0)):
        self.output_size = output_size
        self.scale_range = scale_range

    def __call__(self, sample):
        scale = random.uniform(*self.scale_range)
        crop_size = int(self.output_size[0] * scale), int(self.output_size[1] * scale)
        i, j, h, w = T.RandomCrop.get_params(sample, crop_size)
        sample = TF.crop(sample, i, j, h, w)
        return TF.resize(sample, list(self.output_size), interpolation=TF.InterpolationMode.BILINEAR)


train_transforms = T.Compose([
    RandomScaleCrop(output_size=crop_size, scale_range=(0.75, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])
```

Maybe try Gram Matrix together with VGG
