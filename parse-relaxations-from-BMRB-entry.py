from pynmrstar import Entry
from pynmrstar.saveframe import Saveframe
from pynmrstar.loop import Loop
import numpy as np
import sys, argparse
"""
To take advantage of directing importing from BMRB entries, you will need to install the pynmrstar package
e.g., pip install pynmrstar

For documentation, please read:
- Module: https://pynmrstar.readthedocs.io/en/latest/
- Tags: https://bmrb.io/dictionary/tag.php
"""

def loop_assert(loop, functionName):
    if type(loop) == Saveframe:
        print("= = WARNING: %s() was given a pynmrstar saveframe instead of a loop! Will take the last entry" % functionName, file=sys.stderr )
        loop = loop[-1]
    if type(loop) != Loop:
        print("= = ERROR: %s() was not given a pynmrstar loop instance!" & functionName, file=sys.stderr)
        return None
    else:
        return loop

def is_column_identical(loop,ind):
    l=[ x[ind] for x in loop ]
    l0=l[0]
    for i in l:
        if l0 != i:
            return False
    return True

def get_col_tag_startswith(inp, strStart):
    return [ i for i,tag in enumerate(inp.tags) if tag.startswith(strStart) ]

def get_values_and_errors(loop):
    loop = loop_assert(loop, 'get_values_and_errors')
    name = loop.category
    print("= = Retrieving values from loop %s" % loop.category)
    try:
        val = loop.get_tag('Val')
        err = loop.get_tag('Val_err')
    except:
        print("= = ERROR. Either Val or Val_err tags not found in loop %s; now try to prepend loop name and search again." % loop.category, file=sys.stderr)
        try:
            # = = E.g. What if the values are named T2_val?
            val = loop.get_tag('%s_val' % name.strip('_') )
            err = loop.get_tag('%s_val_err' % name.strip('_') )
        except:
            print("= = ERROR. Appending loop names hasn't worked! Bailing." % loop.category, file=sys.stderr)
            print(loop.tags, file=sys.stderr)
            sys.exit(1)
            
    return val, err

def get_isotopes(loop):
    """
    Given a text data loop, usually the very last one in the save frame, find and return the isotopes for naming.
    Example output: [ '15N', '1H' ]
    """
    # = = = Catch random error in giving the saveFrame instead
    loop = loop_assert(loop, 'get_isotopes')
    # For entries like hetNOE, the tags will be duplicated with
    colsElement    = get_col_tag_startswith(loop, 'Atom_type')
    if np.any( [ loop[0][x] == '.' for x in colsElement] ):
        # = = Entry is incomplete. Get name as backup.
        print("= = WARNING: Entry does not contain Atom_type information. Using Atom_ID as backup.")
        colsElement    = get_col_tag_startswith(loop, 'Atom_ID')

    listElement = [ loop[0][x] for x in colsElement] 
    #print( listElement )
    for c in colsElement:
        if not is_column_identical(loop,c):
            print("= = ERROR in get isotopes(): the column entries for the isotopes are not identical!", file=sys.stderr) 
            return None
    
    colsIsotopeNum = get_col_tag_startswith(loop,'Atom_isotope_number')
    for c in colsIsotopeNum:
        if not is_column_identical(loop,c):
            print("= = ERROR in get isotopes(): the column entries for the isotopes are not identical!", file=sys.stderr) 
            return None

    listIsotopeNum = [ loop[0][x] for x in colsIsotopeNum ]
    if np.any( np.array(listIsotopeNum) == '.' ):
        # = = Entry is incomplete. Get IDs
        print("= = WARNING: Entry does not contain Atom_isotope_number information. Will guess using atom type information.")
        listIsotopeNum = []
        for x in listElement:
            if x == 'H':
                listIsotopeNum.append('1')
            elif x == 'C':
                listIsotopeNum.append('13')
            elif x == 'N':
                listIsotopeNum.append('15')
            elif x == 'O':
                listIsotopeNum.append('17')
            else:
                print("= = ERROR: Atom types is not H C, N, or O. Will bail.", file=sys.stderr)
                sys.exit(1)

    out=[]
    for a,b in zip(listIsotopeNum, listElement):
        out.append( "%s%s" % (a, b) )
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
    inputID   = args.BMRBEntry
    if not inputFile is None:
        entry = Entry.from_file(inputFile)
    elif not inputID is None:
        entry = Entry.from_database(inputID)
    else:
        print("= = ERROR: You must give either an BMRB entry or input file!", file=sys.stderr)
        parser.print_help()
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
        print("    ...loop data tags:", loopData.tags )
        print("    ...loop data first entry:", loopData[0] )
        exptID = d['id']
        sampleCondID = d['sample_condition_list_id']
        freq = d['spectrometer_frequency_1h']
        typeExpt = listTypeExpt[listIndex]
        if not typeExpt == 'NOE':
            exptUnits = d[ listUnitsString[listIndex] ]
            isotopes = get_isotopes(loopData )
            nCols = 19
            #val = [ x[10] for x in loopData ]
            #err = [ x[11] for x in loopData ]
            val,err = get_values_and_errors( loopData )
            #resid   = [ x[4] for x in loopData ]
            #resname = [ x[6] for x in loopData ]
            resid   = loopData.get_tag('Comp_index_ID')
            resname = loopData.get_tag('Comp_ID')
    
            if exptUnits == 's':
                tmp = [ 1./float(x) for x in val ]
                err = [ v*float(e) for v,e in zip(tmp,err) ]
                val = [ float(x) for x in tmp ]
                del tmp
        else:
            exptUnits = ''
            isotopes = get_isotopes(loopData )
            # = = = Change the convention so that the second nuclei is 1H
            if isotopes[0] == '1H':
                isotopes[0] = isotopes[1]
                isotopes[1] = '1H'
            nCols = 33
            val,err = get_values_and_errors( loopData )
            #val = [ x[19] for x in loopData ]
            #err = [ x[20] for x in loopData ]
            tagsResID   = [ t for t in loopData.tags if t.startswith('Comp_index_ID') ]
            tagsResName = [ t for t in loopData.tags if t.startswith('Comp_ID') ]
            resid    = loopData.get_tag(tagsResID[0])
            resname  = loopData.get_tag(tagsResName[0])
            residB   = loopData.get_tag(tagsResID[1])
            resnameB = loopData.get_tag(tagsResName[1])
            #resid    = [ x[4] for x in loopData ]
            #resname  = [ x[6] for x in loopData ]
            #residB   = [ x[13] for x in loopData ]
            #resnameB = [ x[15] for x in loopData ]
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
