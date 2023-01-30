import qcodes

import qcodes_loop._version

__version__ = qcodes_loop._version.__version__


try:
    _register_magic = qcodes.config.core.get('register_magic', False)
    if _register_magic is not False:
        # get_ipython is part of the public api but IPython does
        # not use __all__ to mark this
        from IPython import get_ipython  # type: ignore[attr-defined]

        # Check if we are in IPython
        ip = get_ipython()
        if ip is not None:
            from qcodes.utils.magic import register_magic_class

            register_magic_class(magic_commands=_register_magic)
except ImportError:
    pass
except RuntimeError as e:
    print(e)
