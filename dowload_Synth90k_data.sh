#!/bin/sh

# download s
wget -c http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
tar -xzvf mjsynth.tar.gz

find $PWD | xargs ls -d | grep jpg > image_list.txt
