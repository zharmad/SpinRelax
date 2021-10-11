import numpy as np

# = = = NOTE: Below is a sample PRINT STRIDE FILE data
#! FIELDS time q.w q.x q.y q.z rest0.bias
# 0.000000 0.312824 0.361795 -0.802215 -0.357347 9.982789

# There are N columns, each of which is spaced.
# So, we should read the first field line as a header-designator.
# This needes to be before any data, or else the function should abort.

# It should be noted that PLUMED defaults to single precision.
# Therefore, we are using np.float32 as the conversion.

# This function reads in a PLUMED output done with the PRINT command.
# It expects files of the following form:
# #! FIELDS field1 field2 ...
# entry1 entry2
# entry3 entry4
# ...
#
# The function will check for internal consistency, and return a list of two arrays :
# [ field_names(nfields), parsed_data(nfields, nentries) ]
# The parsed data array is in F-ordering.
def read_from_plumedprint(fname):
    bHeaderRead=False
    #get number of lines and read headers
    ncomm=0
    nempty=0
    ndata=0
    nfields=0
    field_names=[]
    parsed_data=[]
    with open(fname) as fp:
        for ntot, line in enumerate(fp):
            if line == '\n':
                nempty=nempty+1
                continue
            if line.startswith("#"):
                ncomm=ncomm+1
                l = line.split()
                # Check if this comment line is a header line with FIELD.
                if l[1]=="FIELDS":
                    if bHeaderRead:
                        field_names_comp=[ l[i] for i in range(2,len(l)) ]
                        for a,b in zip(field_names, field_names_comp):
                            if a != b:
                                print( '= = ERROR: Multiple FIELD headers are present to indicate parallel trajectoreies, but their entries do not agree!' ) 
                                print( field_names )
                                print( field_names_comp )
                                return -1
                    else:
                        field_names=[ l[i] for i in range(2,len(l)) ]
                        nfields=len(field_names)
                        bHeaderRead=True
                continue

            # The default behaviour is now a data line.
            if not bHeaderRead:
                print( '= = ERROR: Data-like line encountered before a FIELDS definition! Line as follows:' )
                print( line )
                return -1
            l = line.split()
            if len(l) != nfields:
                print( '= = ERROR: Data-like line does not have the same number of fields as defined in FIELDS! ( %i )' % (nfields) )
                print( l )
                return -1
            for i in range(len(l)):
                parsed_data.append(np.float32(l[i]))

    #Add one to enumerate output, then remove counts for comments and empty lines
    ndata=ntot+1-ncomm-nempty
    print( '= = Input file %s has been read: Found %i data-like lines in input plumed FES file, with %i comment lines. ' % (fname, ndata, ncomm) )
    if nempty > 0:
        print( '= = = NOTE: There are %i empty lines' % nempty )
    print( '= = = %i field entries discovered. Field entries are as follows:' % nfields )
    print( str(field_names).strip('[]') )

    # Now reshape parsed_data to match the lines read and the total number of fields.
    parsed_data=np.reshape(parsed_data, (nfields,ndata), order='F')

    return field_names, parsed_data

def read_from_plumedprint_multi(fname):
    bHeaderRead=False
    #get number of lines and read headers
    ncomm=0
    nempty=0
    ndata=0
    nfields=0
    nchunks=0
    field_names=[]
    output_data=[]
    parsed_data=[]
    with open(fname) as fp:
        for ntot, line in enumerate(fp):
            if line == '\n':
                nempty=nempty+1
                continue
            if line.startswith("#"):
                ncomm=ncomm+1
                l = line.split()
                # Check if this comment line is a header line with FIELD.
                if l[1]=="FIELDS":
                    nchunks += 1
                    nf=len(l)
                    temp=[]
                    for i in range(2,nf):
                        temp.append(l[i])
                    nfields=len(temp)
                    field_names.append(temp)
                    bHeaderRead=True
                    #Copy previous data-chunk to a new item
                    if len(parsed_data) != 0:
                        output_data.append(parsed_data)
                        parsed_data=[]
                continue

            # The default behaviour is now a data line.
            if not bHeaderRead:
                print( '= = ERROR: Data-like line encountered before a FIELDS definition! Line as follows:' )
                print( line )
                return -1
            l = line.split()
            if len(l) != nfields:
                print( '= = ERROR: Data-like line does not have the same number of fields as defined in FIELDS!' )
                return -1
            floats=[float(x) for x in l]
            parsed_data.append(floats)

    #Add last bit to overall array.
    output_data.append(parsed_data)

    #Add one to enumerate output, then remove counts for comments and empty lines
    ndata=ntot+1-ncomm-nempty
    print( '= = Input file %s has been read: Found %i data-like lines in input plumed FES file, with %i comment lines. ' % (fname, ndata, ncomm) )
    if nempty > 0:
        print( '= = = NOTE: There are %i empty lines' % nempty )
    print( '= = = %i * %i field entries discovered. Field entries are as follows:' % (nchunks, nfields) )
    print( field_names )

    # Now reshape parsed_data to match the lines read and the total number of fields.
    #parsed_data=np.reshape(parsed_data, (nfields,ndata), order='F')

    return field_names, np.array(output_data)


# This function reverses the read function and outputs a basic PLUMED PRINT file.
#
# It is intended for bug-checking and hacking imitation PLUMED outputs from python.
def write_to_plumedprint(fname, field_names, data):
    # Surrogate print( function, useful for things like mimicking HILLS files and others. )

    #Consistency checks first.
    shape=data.shape
    nfields=len(field_names)
    if shape[0] != nfields:
        print( '= = ERROR: in function write_to_plumedprint, the number of fields do not match between the input data (%i) and field lists (%i)!' % (shape[0], nfields) )
        return -1

    fp=open(fname,'w')
    # Input HEADER data.
    print( "#! FIELDS "+" ".join(field_names), file=fp )

    for i in range(shape[1]):
        print( " ".join("%8f" % data[j][i] for j in range(shape[0])), file=fp )
    fp.close()
    print( '= = File %s has been written.' % fname )
    return
