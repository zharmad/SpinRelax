# SpinRelax

A package to compute NMR spin-relaxation parameters based on molecular dynamics trajectories.

# Change Log and Expected To-Do list

v0.2 - First Refactor code to simplify workflow stages, such as moving bash processing
to within respective python scripts.
 = = = Current = = =
v0.1 - Initial upload of the dirty version using a mixture of
bash, python, PLUMED2, and optionally GROMACS.

# General Information

The package comes with a set of scripts that compute individual stages of the workflow.
This README will provide general information - for more details, please examine
the help documentation for each script.
Hypothetical usages are provided at the end of this document.

Optimally speaking, the user should already possess a centered solute trajectory, in which
the entire tumbling molecule(s) remains unbroken by PBC boundary conditions.
The solute trajectory can then be read by our PLUMED2 fork that computes
quaternion orientations.
While this can be done within PLUMED2, it will not spot mistakes when, say,
a dimer is suddenly split by the PBC into two components.

Alternatively, the quaternion trajectory can be computed by other software
such as the relevant NAMD colvar. In this case, please contact me with
a working example so that I can include code to read the NAMD outputs.

# Overall workflow starting from raw MD trajectory output.

The file
    run-all.bash
contains the overall workflow steps that string together all of the components, such as:
1. center-solute-gromacs.bash (GROMACS)
2. create-reference-pdb.bash (GROMACS)
3. plumed-quat-template.dat (PLUMED)
4. calculate-dq-distribution.py
5. calculate-Ct-from-traj.py
6.  calculate-relaxations-from-Ct.py
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

calculate-relaxations-from-Ct.py then takes the global and local diffusion data,
and combines them into the final spin-relaxation predictions.
Fitting to experiment is done at this stage.

# Example invocations

1. Naive local run.
    run-all.bash -Temp_MD 300 -Temp_Exp 297 -D2O_Exp 0.09 -Bfields 700.303

This is the simple invocation case (and probably most dangerous),
leaving the bash script to look for software/data
and generate all necessary steps, while specifying that the global tumbling
in simulations needs to be tuned from the existing 300K
to match experimental conditions at 297K and 9%-D2O concentration,
and measured at a proton frequency of 700 MHz.

2. Replacing the global tumbling with external knowledge.
    run-all.bash -reffile ../ref-1ubq.pdb -Temp_Exp 297 -D2O_Exp 0.09 -D_ext 3.7383e-5 1.26006 -q_ext 0.866165 0.392069 -0.308123 -0.033159

This is a more likely invocation, where the user may have computed the global tumbling
from HYDRONMR or coarse-grained simulations, or have fitted experimental values from ROTDIF or TENSOR.
An external reference file is supplied, containing the atoms corrresponding to the centered-solute trajectory
and occupancy of 1.0 to ask PLUMED to include this atom in the quaternion orientation frame.

3. Fitting to experimental data
    run-all.bash -reffile ../ref-1ubq.pdb -Temp_Exp 297 -D2O_Exp 0.09 -D_ext 3.7383e-5 1.26006 -q_ext 0.866165 0.392069 -0.308123 -0.033159 -expfile ./expt-R1R2NOE.dat -fit DisoS2

Building up from the previous example, one adds the location of the experimental relaxation file,
and attempts to tune D_iso and zeta-corrections in order to minimise the global chi-deviation.

4. Starting from other programs with a known quaternion trajectory already computed:
    run-all.bash -reffile reference.pdb -qfile colvar-qorient -sxtc solute-centered.dcd

This is a likely starting point for users who possess NAMD trajectories, which will
direct the workflow to skip the initial three steps that are dependent on GROMACS/PLUMED.
The MDTraj plugin should natively recognise common TRAJECTORY files.
However, the workflow will not understand the quaternion colvar output from NAMD at present.
Other than asking me to take into account, please consider formatting the orientation
into the following file format, with no empty lines.
    #! FIELDS time q.w q.x q.y q.z
     0.000000         0.829849         0.293901         0.436507        -0.185565
     10.000000         0.851661         0.273164         0.400721        -0.198689
    ....
     500000.000000         0.148932         0.149478        -0.976888       -0.0341421

5. Some large system with multiple independent MD trajectories
    run-all.bash -folders folder_list -t_mem 50 -reffile /absolute/path/ref.pdb -Temp_MD 300 -Temp_Exp 288 -D2O_Exp 0.10 -D_ext ...(etc.)
    run-all.bash -folders folder_list -t_mem 50 -multiref -reffile ref.pdb -Temp_MD 300 -Temp_Exp 288 -D2O_Exp 0.10 -D_ext ...(etc.)

For many systems, the user will have run multiple simulation copies of the protein
and wish to natively combine the dynamics data from all of them.
This is captured in run-all.bash by providing a file that contains the paths to all folders.
The workflow will attempt to find/generate the required solute trajectory files and quaternion orientations for each folder.
This also includes particular cases where the user has captured details such as alternate protonation
states and need a reference file for each.
Dynamics D-rot and C(t) data from multiple trajectories
are combined by assessing the average values
across all copies at the same time-lag delta-t, rather than averaging the final
values.
Whereas, the ensemble spectral density is still computed per vector N-H recorded in
the solute trajectory, as present in *PhiTheta.dat
