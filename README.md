JINAPyCEE
=========

Public <a href="http://www.jinaweb.org/">JINA</a> Python Chemical Evolution Environment

This repository contains a series of multi-zone galactic chemical evolution codes written in Python 3. It represents an extension of the <a href="http://github.com/NuGrid/NuPyCEE">NuPyCEE</a> package and the end point of <a href="http://www.jinaweb.org/">JINA</a>-<a href="http://www.nugridstars.org/">NuGrid</a> chemical evolution pipeline (see <a href="http://adsabs.harvard.edu/abs/2017nuco.confb0203C">Côté et al. 2017a</a>).

**OMEGA+** (<a href="http://adsabs.harvard.edu/abs/2017arXiv171006442C">Côté et al. 2017b</a>)

- 2-zone model including a galactic (star-forming gas) and circumgalactic (hot gas reservoir) components.
- Star formation based on the balance between gas inflows and stellar feedback.
- Chemical evolution of the galactic component calculated by OMEGA (One-zone Model for the Evolution of GAlaxies, <a href="http://adsabs.harvard.edu/abs/2017ApJ...835..128C">Côté et al. 2017c</a>), a code part of <a href="http://github.com/NuGrid/NuPyCEE">NuPyCEE</a>.

**GAMMA+** (Galaxy Assembly with Merger-trees for Modeling Abundances, <a href="http://adsabs.harvard.edu/abs/2017arXiv171006442C">Côté et al. 2017b</a>)

- Semi-analytic code following the mass assembly of galaxies based on merger trees provided by cosmological simulations.
- Evolution of each building-block galaxy (each branch of the merger tree) calculated by OMEGA+, a code part of JINAPyCEE.

**Requirements**: <a href="http://github.com/NuGrid/NuPyCEE">NuPyCEE</a>. See also our <a href="http://nugrid.github.io/NuPyCEE/">website</a>.

**Userguides**: Coming soon.

**Acknowledgments**: Please cite the references stated above when using codes from this repository.

If you have questions/comments, or want to report a problem, please contact Benoit Côté (<bcote@uvic.ca>, <benoit.cote@csfk.mta.hu>).