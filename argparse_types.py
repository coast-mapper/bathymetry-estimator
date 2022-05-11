import typing
import argparse

from osgeo_setup import *


def coordinate_reference_system(text: str) -> osr.SpatialReference:
    srs_ = osr.SpatialReference()

    methods: typing.List[typing.Callable[[osr.SpatialReference, str], int]] \
        = [osr.SpatialReference.ImportFromWkt,
           osr.SpatialReference.ImportFromProj4]

    for method in methods:
        try:
            method(srs_, text)
            return srs_
        except RuntimeError:
            pass
    try:
        result = srs_.ImportFromEPSG(int(text))
        if result is ogr.OGRERR_NONE:
            return srs_
    except (ValueError, RuntimeError):
        pass

    raise argparse.ArgumentTypeError("Unable to parse \"%s\" as coordinate reference system." % text)


def separated_list(dtype: type = str, sep=',', allowed_values: typing.List = None) -> typing.Callable[
    [str], typing.List]:
    def separated_list_(text: str) -> typing.List[dtype]:
        split = text.split(sep)
        conv = [dtype(e) for e in split]

        if allowed_values is not None:
            for v in conv:
                if v not in allowed_values:
                    raise argparse.ArgumentTypeError("\"%s\" is not allowed." % v)

        return conv

    separated_list_.__name__ = "%s_separated_list" % sep

    return separated_list_


def random_sample_configuration(text: str) -> typing.Tuple[float, int]:
    import time
    parts = text.split(',')

    if len(parts) == 0:
        raise argparse.ArgumentTypeError("\"%s\" - bad random sample configuration." % text)

    try:
        frac = float(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError("Cannot parse \"%s\" as float" % parts[0])

    if len(parts) > 1:
        try:
            state = int(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError("Cannot parse \"%s\" as int" % parts[1])
    else:
        state = int(time.time())

    return frac, state


def separated_tuple(dtypes: typing.List[type], sep=',') -> typing.Callable[[str], typing.Tuple]:
    def separated_tuple_(text: str) -> typing.Tuple:
        split = text.split(sep)

        res = []

        for i in range(len(dtypes)):
            res.append(dtypes[i](split[i]))

        return tuple(res)

    separated_tuple_.__name__ = "%s_separated_tuple" % sep

    return separated_tuple_