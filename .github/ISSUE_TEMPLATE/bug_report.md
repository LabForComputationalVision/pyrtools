---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Please provide a short, reproducible example of the error, for example:
```
import pyrtools as pt
import imageio

img = imageio.imread('DATA/Einstein.pgm'), dtype=torch.float32)
pyr = pt.pyramids.LaplacianPyramid(img)
# this raises an error
recon_img = pyr.recon_pyr()
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**System (please complete the following information):**
 - OS: [e.g. Mac (with version), Ubuntu 18.04]
 - Python version [e.g. 3.7]
 - Pyrtools version [e.g. 1.0.1]

**Additional context**
Add any other context about the problem here.
