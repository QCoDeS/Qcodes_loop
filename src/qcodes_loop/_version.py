def _get_version() -> str:
    from pathlib import Path

    import versioningit

    import qcodes_loop

    module_path = Path(qcodes_loop.__file__).parent
    return versioningit.get_version(project_dir=module_path.parent)


__version__ = _get_version()
