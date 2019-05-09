#!/bin/bash

if [ ! -f hiredis.tar.gz ]; then
    echo "Error! hiredis.tar.gz not found"
    exit 1
fi

if [ ! -d hiredis ]; then
    tar -xvf hiredis.tar.gz
    cd hiredis
    make clean
    make
    make install
    cd ..
else 
    echo "hiredis is already setup!"
fi

if [ ! -f redis-c-cluster.tar.gz ]; then
    echo "Error! redis-c-cluster.tar.gz not found"
    exit 1
fi

if [ ! -d redis-c-cluster ]; then
    tar -xvf redis-c-cluster.tar.gz
    hiredis_dir=$(cd hiredis; pwd)
    cd redis-c-cluster
    sed -i -e "s/^CXXFLAGS=.*/CXXFLAGS=-g -fPIC -Wall -o2 -I $(echo $hiredis_dir/include | sed -e 's/\//\\\//g')/g" Makefile
    sed -i -e "s/^LIB_HIREDIS=.*/LIB_HIREDIS=$(echo $hiredis_dir/lib/libhiredis.a | sed -e 's/\//\\\//g')/g" Makefile
    make clean
    make
    make install
    cd ..
else
    echo "redis-c-cluster is already setup!"
fi 

echo "Done!"