# A 60-min crash course on Gluon

[![Build Status](http://ci.mxnet.io/job/gluon-crash-course/badge/icon)](http://ci.mxnet.io/job/gluon-crash-course/)

Every .md file in this repo may be run directly in Jupyter notebook when you use the `notedown` plugin. Keeping the format in markdown instead of .ipynb makes it easier to collaborate and review on GitHub.

## Prerequisites

You should already have MXNet **1.2.1** or greater installed in a Python virtual environment. Start this environment and continue with Step 1.

### Step 1: Install Jupyter notebook with notedown.

This tarball is provided to do this for you.

```
pip install https://github.com/mli/notedown/tarball/master
```

**Note**: If you run into a problem with this installation at the  `pandoc-attributes` step, run `pip uninstall pypandoc`, then try the installation tarball again.

### Step 2: Start Jupyter with notedown

```
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Before committing updates to these notebooks, please make sure all outputs are cleared.
