#!/bin/bash
# this script uses the ANACONDA_TOKEN env var. 
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/USERNAME/PACKAGENAME --scopes "api:write api:read"
set -e
PACKAGENAME=kinematics

echo "Converting conda package..."
conda convert --platform all $HOME/conda-build/linux-64/${PACKAGENAME}-*.tar.bz2 --output-dir ${HOME}/conda_platforms/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload ${HOME}/conda_platforms/**/${PACKAGENAME}-*.tar.bz2
anaconda -t $ANACONDA_TOKEN upload ${HOME}/conda-build/linux-64/${PACKAGENAME}-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
