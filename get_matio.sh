#!/bin/bash

MATIO_URL=\
http://iweb.dl.sourceforge.net/project/matio/matio/1.5.2/matio-1.5.2.tar.gz
MATIO_VER=matio-1.5.2
INST_DIR=$PWD/matio

if [ -d $INST_DIR ]
then
    echo 'The matio/ directory already exists.'
    echo 'Maybe you already installed matio library already?'
    exit 1
fi
read -p 'Downloading MAT I/O Library. Press any key to continue...'
if [ -f $MATIO_VER.tar.gz ]
then
    rm $MATIO_VER.tar.gz
fi
wget $MATIO_URL
tar xvf $MATIO_VER.tar.gz
mv $MATIO_VER matio
cd matio
./configure --prefix=$INST_DIR
make && make install && rm ../$MATIO_VER.tar.gz
