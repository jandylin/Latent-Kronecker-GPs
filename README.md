# Latent Kronecker GPs
This is the official source code repository for our ICML 2025 paper titled "Scalable Gaussian Processes with Latent Kronecker Structure"

---

To reproduce our experiments:
1. Follow the conda install commands in `install.txt`
2. Set run configuration in `config.yaml`
3. Run the main training script `train.py`

---

The `data/` directory contains the datasets for our experiments:
- `data/sarcos_inv.mat` contains the robotics data for our inverse dynamics prediction experiment, also available at https://gaussianprocess.org/gpml/data/
- `data/LCBench/` contains all 35 individual learning curve datasets from LCBench, also available at https://github.com/automl/LCBench under the Apache-2.0 license
- `data/NGCD_TG.npz` and `data/NGCD_RR.npz` contains the (preprocessed) temperature and precipitation data, respectively, from the [Climate Data Store](https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-nordic), available under the License to Use Copernicus Products. The files contain modified Copernicus Atmosphere Monitoring Service information 2025, and neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

---

If you find this repository useful, please consider citing our paper:
```
@inproceedings{lin2025scalable,  
    title = {Scalable Gaussian Processes with Latent Kronecker Structure}, 
    author = {Jihao Andreas Lin and Sebastian Ament and Maximilian Balandat and David Eriksson and José Miguel Hernández-Lobato and Eytan Bakshy},
    booktitle = {International Conference on Machine Learning},
    year = {2025}
}
```