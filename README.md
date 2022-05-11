# Bathymetry Estimator
## Estymacja batymetrii
Schemat uruchomienia:
```shell script
python predict.py --sentinel-data <Plik wejściowy> --model-dir <katalog z modelem> --result-file <plik wynikowy>
```

Przykładowe uruchomienie:
```shell script
python predict.py --sentinel-data Z:\duze_kafle\S2B_MSIL1C_20200410T100029_N0209_R122_T34UCF_20200410T125625.jp2 --model-dir C:\Projekty\satellitederivedbathymetry\data\models\34 --result-file C:\Projekty\satellitederivedbathymetry\data\bathymetry_estimations\sentinel_34\S2B_MSIL1C_20200410T100029_N0209_R122_T34UCF_20200410T125625.tif
```

Wynik opcji help:
```text
usage: predict.py [-h] --sentinel-data SENTINEL_DATA --model-dir MODEL_DIR
                  [--bathymetry-cutoff BATHYMETRY_CUTOFF]
                  [--preserve-original-size] --result-file RESULT_FILE

optional arguments:
  -h, --help            show this help message and exit
  --sentinel-data SENTINEL_DATA
                        GDAL dataset path for sentinel 1C (default: None)
  --model-dir MODEL_DIR
                        Path model, which be used to estimate bathymetry
                        (default: None)
  --bathymetry-cutoff BATHYMETRY_CUTOFF
                        Value in meters above which batymetry is changed to
                        Inf (default: 16)
  --preserve-original-size
                        With this parameter result image will have same size
                        as input image (default: None)
  --result-file RESULT_FILE
                        Result file location (default: None)
```
## Trenowanie modelu
Przykładowe uruchomienie:
```shell script
python train.py --sentinel-data "Z:\dane_szczecin\2018.vrt" --reference-data "C:\Users\Tomek\Downloads\KW_75974_PM_plik1\Monitoring Wybrzeza dla Politechniki Gdanskiej\2018\data.csv" --reference-data-srs 32633 --test-data-split 0.2 --report-dir Z:\rep\ --model-dir Z:\model\ --gwm-srs 32633 --reference-data-bounds 2,10 --gwm-local-model-range 5000 --gwm-no-of-local-models 15
```

Wyciąg z pomocy:
```text
usage: train.py [-h] --sentinel-data SENTINEL_DATA
                (--reference-data REFERENCE_DATA | --reference-data-folder REFERENCE_DATA_FOLDER)
                [--reference-data-srs REFERENCE_DATA_SRS]
                [--reference-data-bounds REFERENCE_DATA_BOUNDS]
                [--model {GeographyWeightedModel}] [--input-data INPUT_DATA]
                [--disable-gaussian-filtering] [--report-dir REPORT_DIR]
                [--model-dir MODEL_DIR]
                [--model-help {GeographyWeightedModel}]
                [--test-data TEST_DATA | --test-data-split TEST_DATA_SPLIT]
                [--gwm-models-centers-source {compute,predefined}]
                [--gwm-no-of-local-models GWM_NO_OF_LOCAL_MODELS]
                [--gwm-local-model-range GWM_LOCAL_MODEL_RANGE]
                [--gwm-models-centers GWM_MODELS_CENTERS] [--gwm-srs GWM_SRS]

optional arguments:
  -h, --help            show this help message and exit
  --sentinel-data SENTINEL_DATA
                        GDAL dataset path for sentinel 1C (default: None)
  --reference-data REFERENCE_DATA
                        Path to reference data csv file (default: None)
  --reference-data-folder REFERENCE_DATA_FOLDER
                        Path to reference data folder with txt data files
                        (default: None)
  --reference-data-srs REFERENCE_DATA_SRS
                        Optionally you can override default reference data
                        srs. Required is WKT, proj or EPSG numbber (default:
                        None)
  --reference-data-bounds REFERENCE_DATA_BOUNDS
  --model {GeographyWeightedModel}, -m {GeographyWeightedModel}
                        Bathymetry model which will be calibrated (default:
                        GeographyWeightedModel)
  --input-data INPUT_DATA
                        Comma separated list of input data for trained model.
                        Available input data types are: ['x', 'y', 'B2', 'B3',
                        'B4', 'B8', 'raw_bathymery']. Currently following
                        models support free input data setup: None. (default:
                        ['x', 'y', 'raw_bathymery'])
  --disable-gaussian-filtering
  --report-dir REPORT_DIR
                        Path to report directory (default: None)
  --model-dir MODEL_DIR
                        Path where model will be saved (default: None)
  --model-help {GeographyWeightedModel}
                        Displays information about selected model (default:
                        None)
  --test-data TEST_DATA
                        Path to test data csv file (default: None)
  --test-data-split TEST_DATA_SPLIT
                        With this option test data would be extracted from
                        train data. You should provide fraction and optionally
                        random state in such format: frac[,state] . (default:
                        None)

Geography Weighted Model:
  --gwm-models-centers-source {compute,predefined}
  --gwm-no-of-local-models GWM_NO_OF_LOCAL_MODELS
                        Number of local models. Used in case 'compute' option
                        for models center source (default: 12)
  --gwm-local-model-range GWM_LOCAL_MODEL_RANGE
                        Range of local models in meters. (default: 5000)
  --gwm-models-centers GWM_MODELS_CENTERS
                        Model centers in WKT format using MULTIPOINT type.
                        Example: MULTIPOINT ((10 40), (40 30), (20 20), (30
                        10)) (default: None)
  --gwm-srs GWM_SRS     Model centers coordinates system in WKT format.
                        (default: None)
```