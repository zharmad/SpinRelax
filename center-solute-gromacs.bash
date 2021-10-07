#!/bin/bash

function assert_file() {
    [[ $# == 0 ]] && { echo >&2 "= = Nothing given to assert_file! Aborting." ; exit -1 ; }
    while [[ $# -gt 0 ]] ; do
        [ ! -e $1 ] && { echo >&2 "= = File $1 does not exist. Will abort." ; exit -1 ; }
        shift
    done
}

function gmx_type() {
    if type gmx >& /dev/null ; then echo "5.x" ; elif type mdrun >& /dev/null ; then echo "4.x" ; else echo "none" ; fi
}

#Determine GROMACS version
gtyp=$(gmx_type)
case $gtyp in
    5.x)
    gmxsele="gmx select"
    trjconv="gmx trjconv"
    convtpr="gmx convert-tpr"
    ;;
    4.x)
    gmxsele=g_select
    trjconv=trjconv
    convtpr=tpbconv
    ;;
    *)
    echo "= = No GROMACS found! = ="
    exit -1
    ;;
esac

if [ ! $3 ] ; then
    echo "= = (doc) = =
    Usage: ./script <inputTPR> <inputXTC> <outputXTC> [inputNDX]
    Automated Gromacs script for making the solute whole across a trajectory.
    ...If a gromacs index file is not given, look for the first NDX file in the current working directory. 
    ...if the NDX file exists, look for the group \"Solute\" within this file.
    ...If no NDX file exists or no \"Solute\" group present, create a new NDX file called \"solute.ndx\" that excludes waters and ions from the system, assuming that everything else is part of the solute."
    exit
fi
tpr=$1
ixtc=$2
oxtc=$3
workwd=$(pwd)

assert_file $ixtc
assert_file $tpr

if [ $4 ] ; then
    ndx=$4
else
    bMakeNDX=True
    ndx=$(ls $workwd/*.ndx | head -n 1)
    if [[ $ndx != "" ]] ; then
        if grep Solute $ndx ; then
            echo "= = Group Solute found in $ndx - will use this selection."
            bMakNDX=False
        fi
    fi
    if [[ "$bMakeNDX" == "True" ]] ; then
        ndx=./solute.ndx
        echo "= = Making $ndx..."
        $gmxsele -s $tpr -on $ndx -select '"Solute" not group "Water_and_ions"' >& gmx.err || { cat gmx.err >&2 ; exit 1; }
        assert_file $ndx
    fi
if

stpr=./solute.tpr
echo "= = Stage 0/3. Generating solute TPR and compacting."
echo Solute | $convtpr -s $tpr -n $ndx -o $stpr >& gmx.err || { cat gmx.err >&2; exit 1; }
echo "= = Stage 0/3. Compacting system... = ="
echo Solute | $trjconv -s $tpr -n $ndx -f $ixtc -o temp1.xtc -pbc mol -ur compact >& gmx.err  || { cat gmx.err >&2; exit 1; }
echo "= = Stage 1/3 compaction complete... = ="
echo System System System | $trjconv -s $stpr -n $ndx -f temp1.xtc -o temp2.xtc -pbc cluster -ur compact -center  >& gmx.err  || { cat gmx.err >&2; exit 1; }
echo "= = Stage 2/3 clustering and centering complete. = ="
echo System | $trjconv -s $stpr -n $ndx -f temp2.xtc -o $oxtc -pbc mol -ur compact  >& gmx.err  || { cat gmx.err >&2; exit 1; }
echo "= = Stage 3/3 re-clustering complete. Output written to $oxtc = ="
rm -f temp[12].xtc $stpr

# The first step makes protein whole for cluster minimisation algorithm.
# The second step minimises the cluster distance and places the protein at the center. If water is not needed then can stop here.
# The third step re-packs all water molecules around the protein.

