JINAPyCEE
=========

Public <a href="http://www.jinaweb.org/">JINA</a> Python Chemical Evolution Environment

This repository contains a series of multi-zone galactic chemical evolution codes written in Python 3. It represents an extension of the <a href="http://github.com/NuGrid/NuPyCEE">NuPyCEE</a> package (see also our <a href="http://nugrid.github.io/NuPyCEE/">website</a>) and the end point of the <a href="http://www.jinaweb.org/">JINA</a>-<a href="http://www.nugridstars.org/">NuGrid</a> chemical evolution pipeline (see <a href="http://adsabs.harvard.edu/abs/2017nuco.confb0203C">Côté et al. 2017a</a>).

**OMEGA+** (<a href="http://adsabs.harvard.edu/abs/2018ApJ...859...67C">Côté et al. 2018</a>)

- 2-zone model including a galactic (star-forming gas) and circumgalactic (hot gas reservoir) components.
- Star formation based on the balance between gas inflows and stellar feedback.
- Chemical evolution of the galactic component calculated by OMEGA (One-zone Model for the Evolution of GAlaxies, <a href="http://adsabs.harvard.edu/abs/2017ApJ...835..128C">Côté et al. 2017b</a>), a code part of <a href="http://github.com/NuGrid/NuPyCEE">NuPyCEE</a>.

**GAMMA** (Galaxy Assembly with Merger-trees for Modeling Abundances, <a href="http://adsabs.harvard.edu/abs/2018ApJ...859...67C">Côté et al. 2018</a>)

- Semi-analytic code following the mass assembly of galaxies based on merger trees provided by cosmological simulations.
- Evolution of each building-block galaxy (each branch of the merger tree) calculated by OMEGA+, a code part of JINAPyCEE.
- Please see the  <a href="https://github.com/caterpillarproject/caga">caga</a> open-source analysis codes to use GAMMA on top of <a href="http://www.caterpillarproject.org/">Caterpillar</a> merger trees (see <a href="http://adsabs.harvard.edu/abs/2016ApJ...818...10G">Griffen et al. 2016</a>).


**Requirement**

* <a href="http://github.com/NuGrid/NuPyCEE">NuPyCEE</a> to access the chemical evolution functions.
* <a href="https://ytree.readthedocs.io/en/latest/">ytree</a> if you want to read consistent-tree files for GAMMA.

**Userguides**: See the <a href="https://github.com/becot85/JINAPyCEE/tree/master/DOC">DOC</a> directory.

**Acknowledgments**: Please cite the references stated above when using codes from this repository.

If you have questions, comments, or want to report a problem, please contact Benoit Côté (<bcote@uvic.ca>, <benoit.cote@csfk.mta.hu>).

### Installation Instructions

* Create the directory where you want to download the codes.
* Go in that directory with a terminal and clone the GitHub repository.
	* `git clone https://github.com/becot85/JINAPyCEE.git`
* Go into the JINAPyCEE directory and install the codes.
	* `python setup.py develop`
	* **Note**: Use the Python version you will be working with.
* Update the python path to locate JINAPyCEE. This is the path to the directory just before the JINAPyCEE directory, not the one your are currently in.
	* `export PYTHONPATH="your_path_to_before_JINAPyCEE:$PYTHONPATH"`
	* **Important**: Do not forget `:$PYTHONPATH` at the end, otherwise the python path will be overwritten.
	* **Example**: `export PYTHONPATH="benoitcote/gce_code:$PYTHONPATH"`
* **Note**: The `export` command should be put into your bash file. With MAC, it is the .bash_profile file in your home directory. Otherwise, you will need to define the path each time you open a terminal.

* When in Python mode, you can import the code by typing `from JINAPyCEE import omega_plus`, and `from JINAPyCEE import gamma`.
