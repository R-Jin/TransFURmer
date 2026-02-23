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