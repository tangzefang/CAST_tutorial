# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

nbsphinx_allow_errors = True  # Continue through errors
nbsphinx_execute = 'never'   # directly transfer initial ipynb
