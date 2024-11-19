# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.append('/home/unix/panj/wanglab/jessica/CAST_dev')
# sys.path.insert(0, os.path.abspath('/home/unix/panj/wanglab/jessica/CAST/CAST_doc'))
sys.path.insert(0, os.path.abspath('../../api/CAST'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CAST'
copyright = '2024, Zefang Tang; Shuchen Luo; Jessica Pan; Xiao Wang'
author = 'Zefang Tang; Shuchen Luo; Jessica Pan; Xiao Wang'
release = 'Latest'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    "sphinxcontrib.jquery",
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_member_order = 'bysource'
autodoc_inherit_docstrings = False
includehidden = True
collapse_navigation = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

nbsphinx_allow_errors = True  # Continue through errors
nbsphinx_execute = 'never'   # directly transfer initial ipynb
