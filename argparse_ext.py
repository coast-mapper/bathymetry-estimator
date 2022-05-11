import argparse
import warnings
from argparse import ArgumentParser, Namespace
from typing import Union, Text, Sequence, Any, Optional, Tuple


class DeprecatedStoreAction(argparse.Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Union[Text, Sequence[Any], None],
                 *args, **kwargs) -> None:
        warnings.warn(f"Option {self.dest} is deprecated.", category=DeprecationWarning)
        setattr(namespace, self.dest, values)


class ModelHelp(argparse.Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Union[Text, Sequence[Any], None],
                 option_string: Optional[Text] = ...) -> None:
        import models_utils
        if values is not None:
            m_cls = models_utils.get_model_class(values)
            if m_cls is not None:
                models_utils.print_model_description(m_cls)
                parser.exit()
            else:
                parser.exit(-1,"There is no such model: %s" % values)


class SetFigureSizeAction(argparse.Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Optional[Tuple[float, float]],
                 *args, **kwargs) -> None:
        import matplotlib_setup
        matplotlib_setup.set_figure_size(values)