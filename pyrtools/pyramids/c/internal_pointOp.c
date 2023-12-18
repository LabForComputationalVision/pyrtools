#include <stdio.h>
#include <math.h>
#include "internal_pointOp.h"

/* Use linear interpolation on a lookup table.
   Taken from OBVIUS.  EPS, Spring, 1987.
 */
void internal_pointop (im, res, size, lut, lutsize, origin, increment, warnings)
  register double *im, *res, *lut;
  register double origin, increment; 
  register int size, lutsize, warnings;
  {
  register int i, index;
  register double pos;
  register int l_unwarned = warnings;
  register int r_unwarned = warnings;

  lutsize = lutsize - 2;	/* Maximum index value */

  /* printf("size=%d origin=%f lutsize=%d increment=%f\n",size, origin, lutsize,
     increment); */

  if (increment > 0)
    for (i=0; i<size; i++)
      {
	pos = (im[i] - origin) / increment;
	index = (int) pos;   /* Floor */
	if (index < 0)
	  {
	    index = 0;
	    if (l_unwarned)
	      {
		printf("Warning: Extrapolating to left of lookup table...\n");
		l_unwarned = 0;
	      }
	  }
	else if (index > lutsize)
	  {
	    index = lutsize;
	    if (r_unwarned)
	      {
		printf("Warning: Extrapolating to right of lookup table...\n");
		r_unwarned = 0;
	      }
	  }
	res[i] = lut[index] + (lut[index+1] - lut[index]) * (pos - index);
	if(isnan(res[i]))
	  printf("**NAN: lut[%d]=%f lut[%d]=%f pos=%f index=%d\n", index, 
		 lut[index], index+1, lut[index+1], pos, index);
      }
  else
    for (i=0; i<size; i++){
      res[i] = *lut;
      /*printf("res[%d]=%f\n", i, res[i]);*/
    }
  }
