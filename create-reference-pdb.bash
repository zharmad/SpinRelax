#!/bin/bash

function check-ndx() {
    [[ $# -lt 2 ]] && return 1
    grep $1 $2 >& /dev/null
}

function assert_file() {
    [[ $# == 0 ]] && { echo >&2 "= = Nothing given to assert_file! Aborting." ; exit -1 ; }
    while [[ $# -gt 0 ]] ; do
        [ ! -e $1 ] && { echo >&2 "= = File $1 does not exist. Will abort." ; exit -1 ; }
        shift
    done
}

if [ ! $2 ] ; then
    echo "= = (doc) = =
    General Usage: ./script <TPR> <PDB> [NDX]
    This script automagically creates a reference file from the current folder.
    It first looks for a suitable TPR file, and uses it to generate a PDB file using gromacs
    It then looks for an index file solute.ndx, failing that the first *.ndx, if any. Failing this, generate a solute by exclusing all waters and ions according to Gromacs rules."
fi

if type gmx ; then
    echo "= = Found GROMACS 5.x = ="
    gmxsele="gmx select"
    editconf="gmx editconf"
elif type mdrun ; then
    echo "= = Found GROMACS 4.x (or older) = ="
    gmxsele=g_select
    editconf=editconf
else
    echo "= = No GROMACS found! = ="
    exit -1
fi

tpr=$1
assert_file $tpr
opdb=$2

if [ $3 ] ; then
    ndx=$3
else
    bMake=True
    if ls solute.ndx >& /dev/null ; then
        ndx=./solute.ndx
        check-ndx $ndx Solute && bMake=False
    else
        ndx=$(ls *.ndx | head -n 1)
        check-ndx $ndx Solute && bMake=False
    fi

    if [[ "$bMake" == "True" ]] ; then
        ndx=./solute.ndx
        echo "= = No existing Solute group found. Making $ndx..."
        $gmxsele -s $tpr -on $ndx -select '"Solute" not group "Water_and_ions"' >& gmx.err || { cat gmx.err >&2 ; exit 1; }
    else
        echo "= = $ndx exists, and group Solute has been found within it."
    fi
fi

# Create reference molecule with CoG at origin.
echo Solute | $editconf -f $tpr -o $opdb -n $ndx -pbc no -center 0 0 0  >& gmx.err || { cat gmx.err >&2 ; exit 1; }
