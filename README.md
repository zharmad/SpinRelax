# SpinRelax

A workflow to compute NMR spin-relaxation parameters based on molecular dynamics trajectories.

# Change Log and Expected To-Do list

- [x] v0.3  - Fitting to an arbitrary set of spin-relaxation experiments.
              Inclusion of diffusion anistropy based on a single structure instead of a trajectory.
- [x] v0.22 - Residue-specific CSA fittings for one set of R1R2NOE.
- [x] v0.21 - Initial groundwork on residue-specific CSA fittings.
- [x] v0.2  - Port to Python3. Initial work complete with basic validation checks.
- [x] v0.1  - Initial upload of the dirty version using a mixture of bash, python, PLUMED2, and optionally GROMACS.
*(NB: you'll probably have to bug me about individual ticket items.)*
- [ ] Support for eulerian-angle input instead of `-q_ext`.
- [ ] Confirm support for NAMD Quaternion colvars.
- [ ] Try to remove BASH dependency for other users.
- [ ] Make a module to contain the header python scripts.
- [ ] First Refactor code to simplify workflow stages, such as moving bash processing to within respective python scripts.

## Workflow changes as of Jan 2022 over the past year:

- Altered syntax for experimental fitting of data. Conditions such as magnetic fields are included with the epxeirmental data.
- Changed syntax user QoL: center-solute-gromacs.bash 

# General Information

This is the bash/python workflow package associated with Chen, Hologne, Walker and Hennig (2018).
It is designed to compute NMR spin relaxation parameters and fit them to experiment,
based on molecular dynamics simulations data and some estimate of global isotropic tumbling.

The README will provide **general** information only, and 
hypothetical use cases are provided at the end of this document.
Please examine the help documentation for most script by invoking "-h".

## Inputs to the workflow
(1) The user should already possess a centered solute trajectory, in which
the entire tumbling molecule(s) remains unbroken by PBC boundary conditions.
The solute trajectory can then be read by our [PLUMED2 fork](https://github.com/zharmad/plumed2)
that computes quaternion orientations.
While the solute pre-processing can be done entirely within PLUMED2 using **WHOLEMOLECULES**,
the software will not account for edge-cases where, e.g., a dimer is suddenly split by the PBC
into two components.
Thus, far safer for you to verify independently that the trajectory is properly processed.

Alternatively, the quaternion trajectory can be computed by other software
such as the [NAMD](http://www.ks.uiuc.edu/Research/namd) "orientation" collective variable.
In this case, you are free to bug me with a working example
so that I can include code to read the NAMD outputs and cross-check frame definitions.

(2) The majority of molecular forcefields underpredict the viscosity of water solvents,
some of them intentionally so as to expedite sampling. Having an independent estimate, *e.g*.
from [HYDROPRO](http://leonardo.inf.um.es/macromol/programs/hydropro/hydropro.htm) or
[ARMOR-ROTDIF](https://bitbucket.org/kberlin/armor/wiki/Home) will allow you to
skip the quaternion computation step if you are satisfied with structure-based
predictions of global tumbling.

A script **parse-hydroNMR-results.py** is available to help you incorporate this into the workflow.

## Installation

Simply clone into your desired directory, and then run check-installation.bash`.
This will look at whether the shell can find the default arguments and environments required.
Please modify as necessary for your setup - notably, `check-packages.py`
will compile the C-based npufunc and check if the other python scripts can be accessed.

## Integration into popular MD software

...is probably going to be your hard-work. Given the required sampling time
to produce correct results, I don't yet forsee restraints based directly from spin-relaxation.
The codebase is presently easiest to convert into MDTraj, providing within it
several utilities to compute orientation, C(t), and resulting spin-relaxation.

## Citation
Once the paper is accepted, the contents of this repo and link to he paper will be included.

# Overall workflow starting from raw MD trajectory output.

The file `run-all.bash` contains the overall workflow steps
that string together all of the primary components:

1. center-solute-gromacs.bash (GROMACS)
    - This calls GROMACS to pre-process a trajectory to its solute trajectory.
    - KEY output: **solute.xtc** for the complete solute trajectory.
2. create-reference-pdb.bash (GROMACS)
    - This calls GROMACS to create some file representing the reference frame.
    - KEY output: **reference.pdb** for the frames and atoms over which the tumlbing domain is defined.
3. plumed driver -p plumed-quat-template.dat (PLUMED)
    - This calls the PLUMED fork to compute an orientation trajectory.
    - It selects all atoms that have an occupancy > 0.0 to be part of the reference frame.
    - KEY output: **colvar-qorient** for the orientation trajectory.
4. python calculate-dq-distribution.py
    - This reads the solute trajectory to compute global tumbling.
    - It produces the rotational diffusion tensors and quaternion rotations necessary to transform the reference frame into the principal axis frame.
    - NB: The principal axis frame is where the axes of the rotational diffusion tensor D is defined and diagonalised.
    - KEY outputs: **rotdif-iso.dat** for isotropic D, **rotdif-aniso2.dat** for axisymmetric D, **rotdif-aniso_q.dat** for PAF rotations as a function of dt.
5. python calculate-Ct-from-traj.py
    - This reads the solute trajectory to compute local autocorrelation functions, and bond vector distributions.
    - It requires a quaternion rotation operatation if the reference PDB is not already in the principal axis frame,
    - as the output vector distribution is by definition within the PAAF frame and aligned with the diffusion tensor.
    - KEY output: **rotdif_Ctint.dat**, **rotdif_vecHistogram.npz**
6. python calculate-fitted-Ct.py
    - This takes a set of autocorrelations and fits expeonential decay components.
    - KEY output: **rotdif_fittedCt.dat**
7. python calculate-relaxations-from-Ct.py
    - This combines the global and local tumbling factors to prodcue an NMR prediction.
    - [ Deprecated ] Fits to experimental data occur in this step.
    - KEY outputs: **rotdif_R1.dat**,  **rotdif_R2.dat**, **rotdif_NOE.dat**
8. python calculate-relaxations-multi-field.py
    - This rewrite of (7) isused to fits predictions to an arbitrary set of experimental data.
    - KEY outputs: **rotdif_<conditions>_<relaxation>.dat**

The global script runs through each step of the process and checks for the
existance of results from previous invocations.

The optional center-solute-gromacs.bash creates a centered-solute trajectory,
in which the tumbling molecule remains whole within the bounds of the box.
This process is assisted by having one index file (.ndx) that includes the
atom group called "Solute" inside.

create-reference-pdb.bash simply formats and creates the reference file
fo use in downstream applications.

The associated PLUMED2 script run with the separate PLUMED fork produces
the quaternion orientation of the given MD trajections.

calculate-dq-distribution.py and its cousin calculate-dq-distribution-multi.py
gathers the available quaternion orientation from trajectories,
and then computes the relevant global rotational diffusion.

calculate-Ct-from-traj.py separately computes the local motions C(t)
from the centered-solute trajectory.

calculate-fitted-Ct.py converts the raw MD trajectories into a sum of
individual motion parameters, each described by a single exponential.

calculate-relaxations-from-Ct.py then takes the global and local diffusion data,
and combines them into the final spin-relaxation predictions.
Fitting to experiment is done at this stage.

# Example invocations

## 1. *Naive run*
```
run-all.bash -Temp_MD 300 -Temp_Exp 297 -D2O_Exp 0.09 -Bfields 700.303
```
This is a simple invocation case (and probably most dangerous),
leaving the bash script to look for software/data
and generate all necessary steps, while specifying that the global tumbling
in simulations needs to be tuned from the existing 300K
to match experimental conditions at 297K and 9%-D2O concentration,
and measured at a proton frequency of 700 MHz.

## 2. *Replacing the global tumbling with external knowledge*
```
run-all.bash -reffile ../ref-1ubq.pdb -Temp_Exp 297 -D2O_Exp 0.09 -D_ext 3.7383e-5 1.26006 -q_ext 0.866165 0.392069 -0.308123 -0.033159
```
This is a more likely invocation, where the user may have computed the global tumbling
from HYDRONMR or coarse-grained simulations, or have fitted experimental values from ROTDIF or TENSOR.
An external reference file is supplied, containing the atoms corrresponding to the centered-solute trajectory
and occupancy of 1.0 to ask PLUMED to include this atom in the quaternion orientation frame.

## 3. *Fitting to experimental data*
```
run-all.bash -reffile ../ref-1ubq.pdb -D_ext 3.7383e-5 1.26006 -fit Diso,rsCSA -csafile ./initial_CSA_estimates.dat -expfiles ./expt_R1.dat ./expt_R1.dat ./expt_NOE.dat

(NEW SYNTAX) Building up from the previous example, one now adds an list of spin relaxation experiments,
as well as an (optional) list of residue-specific chemical-shift anisotropies.
This particular case asks **calculate-relaxations-multi-field.py** to optimise over
isotropic components of rotational diffusion, and residue-specific CSA.

The CSA argument is given as a file containing two columns of residue-ID followed by residue-specific initial guesses.
Note that the fitting scripts also accept a aingle average valuee as input, and it otherwise defulat to literature values (-170 ppm for 15N.)

To help you set up experimental inputs, please try out **parse-relaxations-from-BMRB-entry.py** with a known spin-relaxation dataset.
E.g., entry 26845 corresponds to Le Masters(2016) and gives R1 and R2 over four spectrometer fields.
The headers of the output is what is required for the input here.

Additional arguments in (2) such as q_ext are not included here; the workflow will
take pre-existing vector distributions file **rotdif_vecHistogram.npz**, which is
defined within the principal-axis frame.

## 4. *Starting from other programs with a known quaternion trajectory already computed*
```
run-all.bash -reffile reference.pdb -qfile colvar-qorient -sxtc solute-centered.dcd
```
This is a likely starting point for users who possess NAMD trajectories, which will
direct the workflow to skip the initial three steps that are dependent on GROMACS/PLUMED.
The MDTraj plugin should natively recognise common TRAJECTORY files.
However, the workflow will not understand the quaternion colvar output from NAMD at present.
Other than asking me to take into account, please consider formatting the orientation
into the following file format, with no empty lines.
```
#! FIELDS time q.w q.x q.y q.z
 0.000000         0.829849         0.293901         0.436507        -0.185565
 10.000000         0.851661         0.273164         0.400721        -0.198689
 ....
 500000.000000         0.148932         0.149478        -0.976888       -0.0341421
```

## 5. *Some large system with multiple independent MD trajectories*
```
run-all.bash -folders folder_list -num_chunks 5 -t_mem 50 ns -reffile /absolute/path/ref.pdb -Temp_MD 300 -Temp_Exp 288 -D2O_Exp 0.10 -D_ext ...(etc.)
run-all.bash -folders folder_list -num_chunks 5 -t_mem 50 ns -multiref -reffile ref.pdb -Temp_MD 300 -Temp_Exp 288 -D2O_Exp 0.10 -D_ext ...(etc.)
```
For many systems, the user will have run multiple simulation copies of the protein
and wish to natively combine the dynamics data from all of them.
This is captured in run-all.bash by providing a file that contains the paths to all folders.
The workflow will attempt to find/generate the required solute trajectory files and quaternion orientations for each folder.
I recommend that all trajectories be of equal length for the sake of the basic error analysis carried out here.
This also includes particular cases where the user has captured details such as alternate protonation
states and need a reference file for each.
Dynamics D-rot and C(t) data from multiple trajectories
are combined by assessing the average values
across all copies at the same time-lag delta-t, rather than averaging the final
values.
Whereas, the ensemble spectral density is still computed per vector N-H recorded in
the solute trajectory, as present in PhiTheta.dat.
Note that the argument -num_chunks will be needed to make sensible error estimates
of the global rotational diffusion: since the error is made by standard deviation over N-subtrajectories,
then it makes no sense to divide a single trjaectory across two chunks. I.e., Have to factorise.

Note that `-t_mem 50 ns sets the **memory time** to 50 ns. This is an subtle but important
idea in NMR, where the MD timescales hidden by global tumbling should be excluded
in the analysis.
