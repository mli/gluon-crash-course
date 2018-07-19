#!/bin/bash
#
# Build and publish all docs

set -x
set -e

# prepare the env
conda env update -f build/build.yml
conda activate gluon_crash_course

pip list

make html
make pkg
make pdf

cp build/_build/latex/gluon_crash_course.pdf build/_build/html/

aws s3 sync --delete build/_build/html/ s3://gluon-crash-course.mxnet.io/ --acl public-read
