#!/usr/bin/gnuplot

# Usage: gnuplot -e "data='FILENAME'" ./script

if (!exists("data")) data='rotdif_vecXH_1.hist'

set size square
unset key
# = = = Greyscale
#set paleete rgbformulae -3,-3,-3
# = = = Heatmap
set palette rgbformulae 21,22,23
set pm3d depthorder
set xyplane at 0

set xlabel "x"
set ylabel "y"
set zlabel "z"

# = = Vector histogram stores data as phi : cos(theta) : count in radians.
splot data u (cos($1)*sqrt(1-$2*$2)):(sin($1)*sqrt(1-$2*$2)):2:3 w pm3d
# = = Alternative if just phi : theta : count
#splot data u (cos($1)*sin($2)):(sin($1)*sin($2)):(cos($2)):3 w pm3d

pause -1
