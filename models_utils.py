import functools
import inspect
import typing
import argparse

import models.abstract


def get_available_models() -> typing.Dict[str, models.abstract.AbstractModelMeta]:
    _available_models = {_m.__name__: _m for _m in
                         models.abstract.AbstractModel.get_implementations()}
    return _available_models


def get_available_non_abstract_models() -> typing.Dict[str, models.abstract.AbstractModelMeta]:
    models = get_available_models()
    return dict(filter(lambda mt: not inspect.isabstract(mt[1]), models.items()))


def get_model_class(name: str) -> typing.Optional[models.abstract.AbstractModelMeta]:
    models = get_available_models()
    if name in models:
        return models[name]
    else:
        return None


class __EllipsisStub(object):
    def __repr__(self):
        return '...'

    __str__ = __repr__


_ellipsis_stub = __EllipsisStub()


def print_model_description(model_cls: models.abstract.AbstractModelMeta):
    meta_data = model_cls.model_class_meta_data
    print("Model name: %s" % model_cls.__name__)
    print("Required input data: %s" % (
        "Any" if meta_data.required_input_data is Ellipsis else
        list(map(lambda e: _ellipsis_stub if e is Ellipsis else e, meta_data.required_input_data))))
    if isinstance(meta_data.required_input_data, list) and Ellipsis in meta_data.required_input_data:
        print("\tThis model has partially bound input data list. To specify input data in place of ... please"
              " use --input-data option while training.")

    print("Does model supports validation data: %s" % ("Yes" if meta_data.supports_validation_data else "No"))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False,
                                     usage=argparse.SUPPRESS)
    model_cls.define_parameters(parser)

    master_cls = model_cls.__base__

    while master_cls != object and master_cls != models.abstract.AbstractModel:
        master_cls.define_parameters(parser)
        master_cls = master_cls.__base__

    help_str = parser.format_help()

    if help_str is not None and len(help_str) > 0:
        print()
        print("Model parameters:")
        print(help_str)


def extract_model_args(model_cls: models.abstract.AbstractModelMeta, args: argparse.Namespace) -> typing.Dict[
    str, typing.Any]:
    prefixes = model_cls.model_class_meta_data.parameter_prefixes

    def is_model_param(name: str):
        result = functools.reduce(lambda a, b: a or b, map(lambda s: name.startswith(s), prefixes), False)
        return result

    ret = {}
    for k in vars(args):
        if is_model_param(k):
            ret[k] = getattr(args, k)

    return ret


def extract_required_input_data_list(model_cls: models.abstract.AbstractModelMeta, args: argparse.Namespace) -> \
        typing.List[str]:
    required_input = model_cls.model_class_meta_data.required_input_data

    if model_cls.model_class_meta_data.required_input_data_has_ellipsis:
        if args.input_data is None:
            raise RuntimeError("Input data should be set for model: %s." % model_cls.__name__)
        if required_input is Ellipsis:
            required_input = args.input_data
        else:
            ret = []
            for e in required_input:
                if e is Ellipsis:
                    ret.extend(args.input_data)
                else:
                    ret.append(e)
            required_input = ret

    return required_input
