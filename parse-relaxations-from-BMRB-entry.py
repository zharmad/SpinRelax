import pynmrstar
import numpy as np
import sys, argparase
"""
To take advantage of directing importing from BMRB entries, you will need to install the pynmrstar package
e.g., pip install pynmrstar
"""

def is_column_identical(loop,ind):
    l=[ x[ind] for x in loop ]
    l0=l[0]
    for i in l:
        if l0 != i:
            return False
    return True

def get_isotopes(loop,colPairs):
    out=[]
    for a,b in colPairs:
        if not is_column_identical(loop,a) or not is_column_identical(loop,b):
            return None
        out.append( "%s%s" % (loop[0][a], loop[0][b]) )
    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script uses the Py-NMRSTAR package to try to '
                                     'read and extract spin relaxation experiments from a BMRB formatted text file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', type=str, dest='BMRBEntry', default=None,
                        help='Download the BMRB entry corresponding to this numeric argument.')
    parser.add_argument('-f', type=str, dest='inputTextFile', default=None,
                        help='Alternatively, use this BMRB-formatted file. This overrules the above argument')
    parser.add_argument('-o', type=str, dest='outputPrefix', default='expt',
                        help='The prefix to all output files. This script will utilise the order and conditions of spin relaxation experiments '
                             'in the input to try to write unique outputs.' )
    args = parser.parse_args()
    outPrefix = args.outputPrefix
    inputFile = args.inputTextFile
    inputID   = args.inputID
    if not inputFile is None:
        entry = pynmrstar.Entry.from_file(inputFile)
    elif not inputIF is None:
        entry = pynmrstar.Entry.from_database(inputID)
    else:
        print("= = ERROR: You must give either an BMRB entry or inptu file!", file=sys.stderr)
        sys.exit(1)

    listInterestedCategories=['heteronucl_T1_relaxation','heteronucl_T2_relaxation','heteronucl_NOEs']
    listUnitsString=['t1_val_units','t2_val_units','']
    listTypeExpt =  ['R1','R2','NOE']

    # = = = Iterate through every save frame until we get to the one containing relaxation data
    count=0 ; listOutputFiles=[]
    for i, saveFrame in enumerate(entry):
        d = saveFrame.tag_dict
        catSF = d['sf_category']
        print("  ...frame %i category: %s" % (i, catSF) )
        if not catSF in listInterestedCategories:
            continue
        print("    ...reading frame data")
        listIndex=listInterestedCategories.index(catSF)
        loopData=saveFrame[-1]
        exptID = d['id']
        sampleCondID = d['sample_condition_list_id']
        freq = d['spectrometer_frequency_1h']
        typeExpt = listTypeExpt[listIndex]
        if not typeExpt == 'NOE':
            exptUnits = d[ listUnitsString[listIndex] ]
            isotopes = get_isotopes(loopData, [(9,8)] )
            nCols = 19
            val = [ x[10] for x in loopData ]
            err = [ x[11] for x in loopData ]
            resid   = [ x[4] for x in loopData ]
            resname = [ x[6] for x in loopData ]
    
            if exptUnits == 's':
                tmp = [ 1./float(x) for x in val ]
                err = [ v*float(e) for v,e in zip(tmp,err) ]
                val = [ float(x) for x in tmp ]
                del tmp
        else:
            exptUnits = ''
            isotopes = get_isotopes(loopData, [(9,8),(18,17)] )
            nCols = 33
            val = [ x[19] for x in loopData ]
            err = [ x[20] for x in loopData ]
            resid    = [ x[4] for x in loopData ]
            resname  = [ x[6] for x in loopData ]
            residB   = [ x[13] for x in loopData ]
            resnameB = [ x[15] for x in loopData ]
        count+=1
        # = = = Avoid assuming that tere are three loops for now, and loop through them all searching for the expected frame
    
        outputFile = '%s_%s_%s_%s_%s.dat' % (outPrefix, typeExpt, freq, exptID, sampleCondID)
        print("    ...writing to file %s" % outputFile )
        fp = open(outputFile, 'w')
        print( "# Type %s" % typeExpt, file=fp)
        print( "# NucleiA %s" % isotopes[0], file=fp)
        if len(isotopes)>1:
            print( "# NucleiB %s" % isotopes[1], file=fp)
        else:
            print( "# NucleiB %s" % '1H', file=fp)
        print( "# Frequency %s" % freq, file=fp)
        print( "# FrequencyUnit MHz", file=fp)
        print( "", file=fp)
        for x, y, dy in zip( resid, val, err):
            print("%s %s %s" % (x,y,dy), file=fp)
        fp.close()
        listOutputFiles.append( outputFile )

    print("= = Finished. %i files written:" % count)
    for x in listOutputFiles:
        print("    %s" % x)
