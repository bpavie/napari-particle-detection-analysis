__version__ = "0.0.1"
from ._widget import ExampleQWidget, example_magic_widget
from ._writer import write_multiple, write_single_image

__all__ = (
    "write_single_image",
    "write_multiple",
    "ExampleQWidget",
    "example_magic_widget",
)
