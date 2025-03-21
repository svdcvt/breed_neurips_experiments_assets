#! /bin/sh
find $1 -type f -name 'event*' | zip $2 -@
