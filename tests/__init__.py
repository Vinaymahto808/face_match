"""Test package.

This ``__init__.py`` is load-bearing: it makes ``tests`` a regular package so it
wins over any unrelated ``tests`` package that happens to sit earlier on
``sys.path`` (there is one in this environment).
"""
