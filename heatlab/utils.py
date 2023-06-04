"""UTILS MODULE"""

#! IMPORTS


from typing import Any, Callable

__all__ = ["check_type", "Signal"]


#! FUNCTIONS


def check_type(obj: object, typ: Any):
    """ensure the object is of the provided type/s"""
    if not isinstance(obj, typ):
        raise TypeError(f"{obj} must be an instance of {typ}.")
    return True


#! CLASSES


class Signal:
    """
    class allowing to generate event signals and connect them to functions.
    """

    # ****** VARIABLES ****** #

    _connected_fun: Callable | None

    # ****** CONSTRUCTOR ****** #

    def __init__(self):
        self._connected_fun = None

    # ****** PROPERTIES ****** #

    @property
    def connected_function(self):
        """return the function connected to this signal."""
        return self._connected_fun

    # ****** METHODS ****** #

    def emit(self, *args, **kwargs):
        """emit the signal with the provided parameters."""
        if self.connected_function is None:
            return None
        elif isinstance(self.connected_function, Callable):
            return self.connected_function(*args, **kwargs)
        else:
            raise TypeError("connected_function is not callable.")

    def connect(self, fun: Callable):
        """
        connect a function/method to the actual signal

        Parameters
        ----------
        fun: FunctionType | MethodType
            the function to be connected to the signal.
        """
        check_type(fun, Callable)
        self._connected_fun = fun

    def disconnect(self):
        """disconnect the signal from the actual function."""
        self._connected_fun = None

    def is_connected(self):
        """check whether the signal is connected to a function."""
        return self.is_connected is not None
