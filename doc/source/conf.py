# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil

notebooks_source = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "notebooks")
)
notebooks_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "notebooks"))

if os.path.exists(notebooks_dest):
    shutil.rmtree(notebooks_dest)
shutil.copytree(notebooks_source, notebooks_dest)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project_version = os.environ.get("DOC_VERSION", "")
project = "HeavyEdge Feature Models"
if project_version:
    project = f"{project} {project_version}"
copyright = "2026, Jisoo Song"
author = "Jisoo Song"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_nb", "sphinx.ext.ifconfig"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"

nb_execution_mode = "off"


def setup(app):
    app.add_config_value("dry_build", os.environ.get("DRY_BUILD", "0") == "1", "env")
