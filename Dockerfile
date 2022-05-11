FROM osgeo/gdal:ubuntu-small-3.4.3

RUN mkdir /bathymetry-estimator
ADD requirements.txt /bathymetry-estimator/
WORKDIR /bathymetry-estimator/

RUN \
    apt-get update && \ 
    apt-get -y install \
	python3-pip && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y python3-pip && \
    apt-get purge python3-pip && \
    apt-get -y autoremove && \
	apt-get clean


ADD *.py /bathymetry-estimator/
COPY models /bathymetry-estimator/models

# ENTRYPOINT [ "/bin/bash" ]
ENTRYPOINT [ "python3.8" ]
CMD predict.py
