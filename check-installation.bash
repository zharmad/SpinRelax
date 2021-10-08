#!/bin/bash

# Check if programs and python modules are present

function assert_cmd() {
    type $1 >/dev/null 2>&1 & { echo >&2 "= = Command $1 verified." ; } || { echo >&2 "= = Command $1 does not exist. Will abort." ; exit -1 ; }
}
function gmx_type() {
    if type gmx >& /dev/null ; then echo "5.x" ; elif type mdrun >& /dev/null ; then echo "4.x" ; else echo "none" ; fi
}


pycmd=python

assert_cmd $pycmd

$pycmd ./check-packages.py

if [ -e ./install_missing_packages.cmd ] ; then
    echo "= = There are missing Python modules that can be installed."
    source ./install_missing_packages.cmd
fi

assert_cmd plumed
gtype=$(gmx_type)
case $gtype in
    5.x) echo "= = GROMACS 5.x installed." ;;
    4.x) echo "= = GROMACS 4.x installed." ;;
    *)   echo "= = GROMACS cannot be found!"
esac
