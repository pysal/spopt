.. Installation

Installation
============

spopt supports Python >= 3.11_. Please make sure that you are
operating in a Python >= 3.11 environment.

Installing released version
---------------------------

spopt is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U spopt

or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

You may also install the latest stable spopt via conda-forge channel by running::

  conda install --channel conda-forge spopt

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of spopt on github - `pysal/spopt`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/spopt`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/spopt.git

You can  also `fork`_ the `pysal/spopt`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/spopt`_, you can
contribute to spopt development.

.. _3.11: https://docs.python.org/3.11/
.. _Python Package Index: https://pypi.org/project/spopt/
.. _pysal/spopt: https://github.com/pysal/spopt
.. _fork: https://help.github.com/articles/fork-a-repo/ 

