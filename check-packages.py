import sys, os

script_loc=os.path.dirname(os.path.realpath(__file__))
#script_loc=os.path.abspath(__file__)
pycmd=sys.executable
fname="install_missing_modules.cmd"
print "= = Running module check."
module_list=['numpy', 'scipy', 'mdtraj', 'transforms3d']
version_rec=[ '1.10.0', '0.17.1', '1.7.2', '0.0']
modules = map(__import__, module_list)
num_modules = len(module_list)
bLacking=False

python_ver = sys.version_info
if sys.version_info[0] < 2 or sys.version_info[1] < 7:
    print >> sys.stderr, "= = ERROR: Python 2.7+ needed."
    sys.exit(1)
else:
    print "= = Found Python %s" % sys.version

import importlib

fp = open(fname, 'w')
print >> fp, "cd %s" % script_loc
for i in range(num_modules):
    name=module_list[i]
    try:
        importlib.import_module(name)
        print "    ... %s is installed." % name
        try:
            v=modules[i].__version__
        except:
            try:
                v=modules[i].version.version
            except:
                v='unknown'
        if v != 'unknown':
            print "    ... Version Installed: %s, recommended >= %s" % (v, version_rec[i])
        else:
            print "    ... Version is unknown."

    except ImportError:
        bLacking=True
        print "    ... %s is not installed. Adding pip command to install instructions." % name
        print >> fp, "sudo pip install %s" % name

# = = C-coded universal function to handle J_omega quickly.
try:
    importlib.import_module('npufunc')
    print "    ... The numpy universal function for J_omega computation is installed."
except ImportError:
    bLacking=True
    print "    ... The numpy universal function for J_omega computation is not installed. Adding its install script."
    print >> fp, "source ufunc_Jomega/install.sh"

# = = Other standard functions that should be in PYTHONPATH"
try:
    i = importlib.import_module('spectral_densities')
    print "    ... spectral_densities routines found. PYTHONPATH seems to be set correctly."
except ImportError:
    bLacking=True
    print "    ... spectral_densities routines not found! PYTHONPATH is probably not set correctly."
    print >> fp, "export PYTHONPATH=%s:$PYTHONPATH" % script_loc
fp.close()

if bLacking:
    print "= = module check encountered missing modules. Please check/run %s to install them." % fname
else:
    print "= = All modules present."
    os.remove(fname)
