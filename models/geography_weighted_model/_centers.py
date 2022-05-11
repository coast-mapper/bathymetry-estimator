from typing import Union

import numpy as np
import pandas as pd
import sklearn.cluster


def calculate_local_centers(data: Union[np.ndarray, pd.DataFrame], no_of_centers: int) -> pd.DataFrame:
    kmeans = sklearn.cluster.KMeans(no_of_centers)
    kmeans.fit(data)
    return pd.DataFrame(kmeans.cluster_centers_, columns=["x", "y"])