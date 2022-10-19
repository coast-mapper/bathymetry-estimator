# Bathymetry Estimator
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6779671.svg)](https://doi.org/10.5281/zenodo.6779671)

This software is designed to train, evaluate and run bathymetry estimation model which uses earth observation data from Sentinel-2 satellites as input.
It consists of 3 main scripts:

* `train.py` - which is used to train model using Sentinel-2 images and reference data acquired by in-situ measurements,
* `evaluate.py` - this script is used to quality of estimations made by specific model on provided data. Especially on other non-training data.
* `predict.py` - this script is indented for operational use. It predicts bathymetry using input image with provided model.

This software supports following models:

* Geography weighed model _[1]_
* Neural network model based implemented with Keras engine _[2]_,
* Simple linear model based implemented with Keras engine _[2]_,
* Geography weighed model _[1]_, implemented with Keras engine _[2]_,
* Geography weighed model _[1]_ with neural networks, implemented with Keras engine _[2]_,
* Decision tree regression using sklearn library _[3]_,
* Random forest regression using sklearn library _[3]_,

## Software setup

All required Python libraries are listed in `requirements.txt`.
Installation and setup of those libraries can be hard, especially on Windows. This problem is caused by GDAL library, which fails to install on Windows via `pip` command (as of May 17, 2022).

There are three tested methods of setting up this software:
* On Windows: by creative configuration of virtual environment (venv) involving [OSGeo4w](https://www.osgeo.org/projects/osgeo4w/)
* Using Docker (recommended): by building docker image using `Dockerfile` present in this repository,
* On Linux: by installing several libraries (including GDAL) with Python bindings from distribution packages in similar way like it is described in `Dockerfile` (Note that docker image is build using base image with pre-installed GDAL with Python bindings). 

### Building docker image

In main directory please execute following command:
```shell
docker build -t bathymetry-estimator .
```

Above command will finish after several minutes. It will create docker image named `bathymetry-estimator` in local docker registry. 
Apparent size of image would be 2.5 GB.

To test build image please run following command:
```shell
docker run --rm bathymetry-estimator
```

it should result with output similar to:

```text
usage: predict.py [-h] --sentinel-data SENTINEL_DATA
                  [--disable-gaussian-filtering] --model-dir MODEL_DIR
                  [--bathymetry-cutoff BATHYMETRY_CUTOFF]
                  [--preserve-original-size] --result-file RESULT_FILE
                  [--mask-file MASK_FILE]
                  [--operation-tile-sizes OPERATION_TILE_SIZES]
predict.py: error: the following arguments are required: --sentinel-data, --model-dir, --result-file
```

Other scripts can be run using command similar to:

```shell
docker run --rm bathymetry-estimator evaluate.py -h
```

Above command should print help for `evaluate.py` script.

### Running docker image with data

Let's assume that there is directory `data` located in current working directory.
To run docker image with mounted data please construct command in following way:

Powershell on Windows
```powershell
docker run --rm -v "${pwd}/data:/data" -ti bathymetry-estimator <script_name> <script_parameters>
```

Bash on Linux
```shell
docker run --rm -v `pwd`/data:/data -ti bathymetry-estimator <script_name> <script_parameters>
```

## Assumptions 
For rest of this Readme file location of data directory will be assumed as `/data` (This supports above docker configuration).
Also in case of docker configuration please precede every mentioned command with proper docker prefix.

```powershell
docker run --rm -v "${pwd}/data:/data" -ti bathymetry-estimator
```

or

```shell
docker run --rm -v `pwd`/data:/data -ti bathymetry-estimator
```

Alternatively if script is run in properly configured python environment please precede commnads with
```shell
python
```

and replace `/data` with proper path to dataset.


## Training model

### Running train script

Following train examples will be run this dataset: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6543997.svg)](https://doi.org/10.5281/zenodo.6543997)


#### Common options

To display help command please run

```shell
train.py -h
```

This will print quite long output which includes description of every parameter for every model.
Important part of this output is description of common options:
```text
  -h, --help            show this help message and exit
  --sentinel-data SENTINEL_DATA
                        GDAL dataset path for sentinel (default: None)
  --disable-gaussian-filtering
  --reference-data REFERENCE_DATA
                        Path to reference data csv file (default: None)
  --reference-data-folder REFERENCE_DATA_FOLDER
                        Path to reference data folder with txt data files (default: None)
  --reference-data-srs REFERENCE_DATA_SRS
                        Optionally you can override default reference data srs. Required is WKT, proj or EPSG number (default: None)
  --reference-data-bounds REFERENCE_DATA_BOUNDS
  --model {LinearModel,NeuralNetworkModel,RegressionForestModel,GeographyWeightedModel,GeographyWeightedLinearModel,RegressionTreeModel,GeographyWeightedNeuralNetworkModel}, -m {LinearModel,NeuralNetworkModel,RegressionForestModel,GeographyWeightedModel,GeographyWeightedLinearModel,RegressionTreeModel,GeographyWeightedNeuralNetworkModel}
                        Bathymetry model which will be calibrated (default: GeographyWeightedModel)
  --input-data INPUT_DATA
                        Comma separated list of input data for trained model. Available input data types are: ['x', 'y', 'B2', 'B3', 'B4', 'B8', 'raw_bathymetry']. Currently following models support free input data setup:
                        ['LinearModel', 'NeuralNetworkModel', 'RegressionForestModel', 'RegressionTreeModel', 'GeographyWeightedNeuralNetworkModel']. (default: ['x', 'y', 'raw_bathymetry'])
  --report-dir REPORT_DIR
                        Path to report directory (default: None)
  --model-dir MODEL_DIR
                        Path where model will be saved (default: None)
  --model-help {LinearModel,NeuralNetworkModel,RegressionForestModel,GeographyWeightedModel,GeographyWeightedLinearModel,RegressionTreeModel,GeographyWeightedNeuralNetworkModel}
                        Displays information about selected model (default: None)
  --validation-data-split VALIDATION_DATA_SPLIT
                        With this option validation data would be extracted from train data. You should provide fraction and optionally random state in such format: frac[,state] . (default: None)
  --test-data TEST_DATA
                        Path to test data csv file (default: None)
  --test-data-split TEST_DATA_SPLIT
                        With this option test data would be extracted from train data. You should provide fraction and optionally random state in such format: frac[,state] . (default: None)
  --normalize-input-data
```

Those options could be divided in following groups:

Training data location and description:
* `--sentinel-data` (can be specified multiple times),
* `--reference-data` or `--reference-data-folder`,
* `--reference-data-srs`,
* `--test-data` (alternatively `--test-data-split` can be used)

Input data preprocessing:
* `--disable-gaussian-filtering` - disables 3x3 gaussian filter applied to Sentinel data,
* `--reference-data-bounds` - cuts reference data set to specified depth range,
* `--normalize-input-data` - normalization of input data - very useful in case of neural networks

Input data split to subsets:
* `--test-data-split`,
* `--validation-data-split`

Model common options:
* `-m` or `--model`,
* `--input-data` (if supported),
* `--model-dir`,
* `--report-dir` (optionally)

Model specific options: see model help.

#### Model help

To print help for specific model type please use following command:

```shell
train.py --help-model <model_name>
```

for instance

```shell
train.py --model-help GeographyWeightedModel
```

which should print output similar to:
```text
Model name: GeographyWeightedModel                                                                                                                                                                                                      
Required input data: ['x', 'y', 'raw_bathymetry']         
Does model support validation data: No

Model parameters:
Geography Weighted Model:
  --gwm-mode {linear,exponential}
                        Type of regression used for local models. For linear model is in form a*x+b, for exponential is exp(a*x+b). (default: linear)
  --gwm-models-centers GWM_MODELS_CENTERS
                        Model centers in WKT format using MULTIPOINT type. Example: MULTIPOINT M((10 40 300), (40 30 300), (20 20 300), (30 10 400)) (default: None)
  --gwm-srs GWM_SRS     Model centers coordinates system in WKT format. (default: None)
  --gwm-models-centers-source {compute,predefined}
                        How models centers are determined. Compute is deprecated (default: predefined)
  --gwm-no-of-local-models GWM_NO_OF_LOCAL_MODELS
                        Number of local models. Used in case 'compute' option for models center source (default: None)
  --gwm-local-model-range GWM_LOCAL_MODEL_RANGE
                        Range of local models in meters. (default: None)
```

#### Sample training command

Following train examples will be run this dataset: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6543997.svg)](https://doi.org/10.5281/zenodo.6543997)

For instance, to train Geography weighted model following command can be used:

```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwm --model-dir /data/model_gwm --gwm-models-centers-source compute --gwm-no-of-local-models 50 --gwm-local-model-range 2500
```

Meaning of parameters is following:
* `--reference-data-bounds 0,16` - from in situ measurements take only those in range from 0 m to 16 m,
* `-m GeographyWeightedModel` - chosen model is `GeographyWeightedModel`,
* `--sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2` - location of sentinel-2 data raster,
* `--reference-data /data/reference_data_34.csv` - location of reference data,
* `--test-data-split 0.1,2` - test data will be extracted as 10% training data in pseudo random process. Number `2` is provided as seed to this process,
* `--report-dir /data/report_gwm` - location of report directory which will contain information about training process,
* `--model-dir /data/model_gwm ` - path to directory in which model will be saved. This path should be given as parameter to `predict.py` and `evaluate.py` scripts,
* `--gwm-models-centers-source compute --gwm-no-of-local-models 50 --gwm-local-model-range 2500` - `GeographyWeightedModel` specific options.

## Prediction using trained model

To display help for predict script please use following command:
```shell
predict.py --help
```

This is sample predict command:
```shell
predict.py --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --model-dir /data/model_gwm --result-file /data/estimation.tif
```

## Reference:

* [1] [Andrzej Chybicki (2018) Three-Dimensional Geographically Weighted Inverse Regression (3GWR) Model for Satellite Derived Bathymetry Using Sentinel-2 Observations, Marine Geodesy, 41:1, 1-23, DOI: 10.1080/01490419.2017.1373173](https://www.tandfonline.com/doi/abs/10.1080/01490419.2017.1373173?journalCode=umgd20)
* [2] [Chollet, F., & others. (2015). Keras. https://keras.io.](https://keras.io),
* [3] [Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)


## Appendix: Sample model training commands

Following train examples will be run this dataset: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6543997.svg)](https://doi.org/10.5281/zenodo.6543997)


Geography weighed model *[ref]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwm --model-dir /data/model_gwm --gwm-models-centers-source compute --gwm-no-of-local-models 50 --gwm-local-model-range 2500
```

Geography weighed model in exponential mode *[exp]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwm_exp --model-dir /data/model_gwm_exp --gwm-models-centers-source compute --gwm-no-of-local-models 50 --gwm-local-model-range 2500 --gwm-mode exponential 
```

Geography weighed model Keras implementation (simultaneous mode) *[gwmk]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedLinearModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwmk --model-dir /data/model_gwmk --gwmk-models-centers "MULTIPOINT M ((346637.758441 6067158.990684 5000),(316095.21729 6080528.379283 5000),(355724.469898 6060259.989847 5000),(337819.139844 6072679.203969 5000),(304999.063218 6080759.581897 5000),(330140.222103 6078504.368026 5000),(358940.068829 6056532.706564 5000),(322575.195312 6080311.178385 5000),(351325.322947 6064048.656232 5000),(342171.990224 6070057.40372 5000),(335050.704144 6074917.766471 5000),(310356.334711 6080754.661157 5000))" --gwmk-srs 32634 --validation-data-split 0.2,2 --keras-max-iterations 1000
```

Geography weighed model Keras implementation (separate mode) *[gwmk_ref]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedLinearModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwmk_ref --model-dir /data/model_gwmk_ref --gwmk-models-centers "MULTIPOINT M ((346637.758441 6067158.990684 5000),(316095.21729 6080528.379283 5000),(355724.469898 6060259.989847 5000),(337819.139844 6072679.203969 5000),(304999.063218 6080759.581897 5000),(330140.222103 6078504.368026 5000),(358940.068829 6056532.706564 5000),(322575.195312 6080311.178385 5000),(351325.322947 6064048.656232 5000),(342171.990224 6070057.40372 5000),(335050.704144 6074917.766471 5000),(310356.334711 6080754.661157 5000))" --gwmk-srs 32634 --validation-data-split 0.2,2 --keras-max-iterations 15 --gwmk-train-mode separate
```

Neural networks 10 tanh, 1 lin in GWM architecture. Inputs: qlog (raw_bathymetry) *[gwmk_nn]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedNeuralNetworkModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwmknn --model-dir /data/model_gwmknn --gwmk-models-centers "MULTIPOINT M ((346637.758441 6067158.990684 5000),(316095.21729 6080528.379283 5000),(355724.469898 6060259.989847 5000),(337819.139844 6072679.203969 5000),(304999.063218 6080759.581897 5000),(330140.222103 6078504.368026 5000),(358940.068829 6056532.706564 5000),(322575.195312 6080311.178385 5000),(351325.322947 6064048.656232 5000),(342171.990224 6070057.40372 5000),(335050.704144 6074917.766471 5000),(310356.334711 6080754.661157 5000))" --gwmk-srs 32634 --validation-data-split 0.2,2 --keras-max-iterations 1000 --gwmknn-layer 10,tanh --gwmknn-layer 1,linear --input-data raw_bathymetry
```

Neural networks 10 tanh, 1 lin in GWM architecture. Inputs: B2, B3, B4, B8 *[gwmk_nn_bands]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedNeuralNetworkModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwmknn_bands --model-dir /data/model_gwmknn_bands --gwmk-models-centers "MULTIPOINT M ((346637.758441 6067158.990684 5000),(316095.21729 6080528.379283 5000),(355724.469898 6060259.989847 5000),(337819.139844 6072679.203969 5000),(304999.063218 6080759.581897 5000),(330140.222103 6078504.368026 5000),(358940.068829 6056532.706564 5000),(322575.195312 6080311.178385 5000),(351325.322947 6064048.656232 5000),(342171.990224 6070057.40372 5000),(335050.704144 6074917.766471 5000),(310356.334711 6080754.661157 5000))" --gwmk-srs 32634 --validation-data-split 0.2,2 --keras-max-iterations 1000 --gwmknn-layer 10,tanh --gwmknn-layer 1,linear --input-data B2,B3,B4,B8
```

Neural networks 10 tanh, 1 lin in GWM architecture. Inputs: qlog (raw_bathymetry), B2, B3, B4, B8 *[gwmk_nn_bands_qlog]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedNeuralNetworkModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwmknn_bands_qlog --model-dir /data/model_gwmknn_bands_qlog --gwmk-models-centers "MULTIPOINT M ((346637.758441 6067158.990684 5000),(316095.21729 6080528.379283 5000),(355724.469898 6060259.989847 5000),(337819.139844 6072679.203969 5000),(304999.063218 6080759.581897 5000),(330140.222103 6078504.368026 5000),(358940.068829 6056532.706564 5000),(322575.195312 6080311.178385 5000),(351325.322947 6064048.656232 5000),(342171.990224 6070057.40372 5000),(335050.704144 6074917.766471 5000),(310356.334711 6080754.661157 5000))" --gwmk-srs 32634 --validation-data-split 0.2,2 --keras-max-iterations 1000 --gwmknn-layer 10,tanh --gwmknn-layer 1,linear --input-data raw_bathymetry,B2,B3,B4,B8
```

Neural networks 10 tanh, 1 exp in GWM architecture. Inputs: qlog (raw_bathymetry) *[gwmk_nn_exp]*
```shell
train.py --reference-data-bounds 0,16 -m GeographyWeightedNeuralNetworkModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_gwmknn_exp --model-dir /data/model_gwmknn_exp --gwmk-models-centers "MULTIPOINT M ((346637.758441 6067158.990684 5000),(316095.21729 6080528.379283 5000),(355724.469898 6060259.989847 5000),(337819.139844 6072679.203969 5000),(304999.063218 6080759.581897 5000),(330140.222103 6078504.368026 5000),(358940.068829 6056532.706564 5000),(322575.195312 6080311.178385 5000),(351325.322947 6064048.656232 5000),(342171.990224 6070057.40372 5000),(335050.704144 6074917.766471 5000),(310356.334711 6080754.661157 5000))" --gwmk-srs 32634 --validation-data-split 0.2,2 --keras-max-iterations 1000 --gwmknn-layer 10,tanh --gwmknn-layer 1,linear --input-data raw_bathymetry
```

Decision tree regression. Inputs: B2, B3, B4, B8 *[rt_bands]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionTreeModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rt_bands --model-dir /data/model_rt_bands --input-data B2,B3,B4,B8
```

Decision tree regression. Inputs: qlog (raw_bathymetry) *[rt_qlog]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionTreeModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rt_qlog --model-dir /data/model_rt_qlog --input-data raw_bathymetry
```

Decision tree regression. Inputs: qlog (raw_bathymetry), B2, B3, B4, B8 *[rt_qlog_bands]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionTreeModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rt_qlog_bands --model-dir /data/model_rt_qlog_bands --input-data raw_bathymetry,B2,B3,B4,B8
```

Decision tree regression. Inputs: x, y, B2, B3, B4, B8 *[rt_spatial_bands]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionTreeModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rt_spatial_bands --model-dir /data/model_rt_spatial_bands --input-data x,y,B2,B3,B4,B8
```

Decision tree regression. Inputs: x, y, qlog (raw_bathymetry) *[rt_spatial_qlog]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionTreeModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rt_spatial_qlog --model-dir /data/model_rt_spatial_qlog --input-data x,y,raw_bathymetry
```

Decision tree regression. Inputs: x, y, qlog (raw_bathymetry), B2, B3, B4, B8 *[rt_spatial_qlog_bands]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionTreeModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rt_spatial_qlog_bands --model-dir /data/model_rt_spatial_qlog_bands --input-data x,y,raw_bathymetry,B2,B3,B4,B8
```

Random forest regression. Inputs: B2, B3, B4, B8 *[rf_bands]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionForestModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rf_bands --model-dir /data/model_rf_bands --input-data B2,B3,B4,B8
```

Random forest regression. Inputs: qlog (raw_bathymetry) *[rf_qlog]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionForestModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rf_qlog --model-dir /data/model_rf_qlog --input-data raw_bathymetry
```

Random forest regression. Inputs: qlog (raw_bathymetry), B2, B3, B4, B8 *[rf_qlog_bands]*
```shell
train.py --reference-data-bounds 0,16 -m RegressionForestModel --sentinel-data /data/S2A_MSIL1C_20190630T100031_N0207_R122_T34UCF_20190630T120400.jp2 --reference-data /data/reference_data_34.csv --test-data-split 0.1,2 --report-dir /data/report_rf_qlog_bands --model-dir /data/model_rf_qlog_bands --input-data raw_bathymetry,B2,B3,B4,B8
```