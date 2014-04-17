#include <Python.h>
/* #include "Numeric/arrayobject.h" */
#include "numpy/arrayobject.h"
#include "convolve.h"

/* compile with: gcc -shared -I/usr/include/python2.6/ -lpython2.6 -o pyPyrCcode.so -fPIC pyPyrCcode.c convolve.c convolve.h edges.c wrap.c internal_pointOp.c*/

#define notDblMtx(it) (!mxIsNumeric(it) || !mxIsDouble(it) || mxIsSparse(it) || mxIsComplex(it))


static PyObject* py_corrDn(PyObject* self, PyObject* args)
{
  int x_fdim, y_fdim;
  int x_idim, y_idim;
  PyObject *arg1, *arg2;
  PyArrayObject *image, *filt;
  PyArrayObject *result = NULL;
  char *edges = "reflect1";
  int x_step = 1;
  int y_step = 1;
  int x_start = 0;
  int y_start = 0;
  int x_stop, y_stop, x_rdim, y_rdim;
  int dimensions[1];
  int i;

  // parse input parameters
  if( !PyArg_ParseTuple(args, "iiOiiO|siiiiiiO", &x_idim, &y_idim, &arg1, 
			&x_fdim, &y_fdim, &arg2, &edges, &x_step, &y_step, 
			&x_start, &y_start, &x_stop, &y_stop, &result) )
    return NULL;
  image = (PyArrayObject *)PyArray_ContiguousFromObject(arg1, PyArray_DOUBLE, 1,
							x_idim * y_idim);

  x_stop = x_idim;
  y_stop = y_idim;
  if(image == NULL)
    return NULL;
  /*if(image->nd != 2 || image->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type double");
    return NULL;
    }*/

  filt = (PyArrayObject *)PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 
						       1, x_fdim * y_fdim);
  if(filt == NULL)
    return NULL;
  if(filt->nd != 2 || filt->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type int");
    return NULL;
  }

  if( (x_fdim > x_idim) || (y_fdim > y_idim) ){
    printf("Filter: [%d %d], Image: [%d %d]\n", x_fdim, y_fdim, x_idim, y_idim);
    printf("FILTER dimensions larger than IMAGE dimensions.\n");
    exit(1);
  }

  if ( (x_step < 1) || ( y_step < 1) ){
    printf("STEP values must be greater than zero.");
    exit(1);
  }

  if ( (x_start < 0) || (x_start > x_idim) || 
       (y_start < 0) || (y_start > y_idim) ){
    printf("START values must lie between 1 and the image dimensions.");
    exit(1);
  }

  if ( (x_stop < x_start) || (x_stop > x_idim) ||
       (y_stop < y_start) || (y_stop > y_idim) ){
    printf("STOP values must lie between START and the image dimensions.");
    exit(1);
  }

  x_rdim = (x_stop-x_start+x_step-1) / x_step;
  y_rdim = (y_stop-y_start+y_step-1) / y_step;

  //dimensions[0] = (x_idim/y_step) * (y_idim/x_step);
  dimensions[0] = x_rdim * y_rdim;

  if(result == NULL)
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_DOUBLE);

  double *temp = malloc(x_fdim * y_fdim * sizeof(double));
  if (temp == NULL){
    printf("Cannot allocate necessary temporary space");
    exit(1);
  }

  printf("i(%d, %d), f(%d, %d), r(%d, %d), X(%d, %d, %d), Y(%d, %d, %d),%s\n",
	 x_idim,y_idim,x_fdim,y_fdim,x_rdim,y_rdim,
	 x_start,x_step,x_stop,y_start,y_step,y_stop,edges);
    
  if (strcmp(edges,"circular") == 0)
    internal_wrap_reduce((image_type *)image->data, x_idim, y_idim, 
			 (image_type *)filt->data, x_fdim, y_fdim,
  			 x_start, x_step, x_stop, y_start, y_step, y_stop,
  			 (image_type *)result->data);
  else 
    internal_reduce((image_type *)image->data, x_idim, y_idim, 
		    (image_type *)filt->data, temp, x_fdim, y_fdim,
		    x_start, x_step, x_stop, y_start, y_step, y_stop,
		    (image_type *)result->data, edges);
  
  free(temp);

  return PyArray_Return(result);
} 


static PyObject* py_upConv(PyObject* self, PyObject* args)
{
  int x_fdim, y_fdim, x_idim, y_idim;
  PyObject *arg1, *arg2; 
  PyObject *arg3 = NULL;
  PyArrayObject *image, *filt, *orig_filt, *new_filt;
  PyArrayObject *result = NULL;
  int orig_x = 0;
  int orig_y, x, y;
  char *edges = "reflect1";
  int x_step = 1;
  int y_step = 1;
  int x_start = 0;
  int y_start = 0;
  int x_stop, y_stop, x_rdim, y_rdim;
  int dimensions[1];
  int dimensions2[1];
  int dimensions3[1];
  int i;

  // parse input parameters
  if( !PyArg_ParseTuple(args, "iiOiiO|siiiiiiO", &x_idim, &y_idim, &arg1, 
			&x_fdim, &y_fdim, &arg2, &edges, &x_step, &y_step, 
			&x_start, &y_start, &x_stop, &y_stop, &arg3) )
    return NULL;

  image = (PyArrayObject *)PyArray_ContiguousFromObject(arg1, PyArray_DOUBLE, 1,
							x_idim * y_idim);
  
  //x_stop = x_idim;
  //y_stop = y_idim;
  if(image == NULL)
    return NULL;
  if(image->nd != 2 || image->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type double\n");
    return NULL;
  }

  filt = (PyArrayObject *)PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 
						       1, x_fdim * y_fdim);
  if(filt == NULL)
    return NULL;
  if(filt->nd != 2 || filt->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type double\n");
    return NULL;
  }

  /*if( (x_fdim > x_idim) || (y_fdim > y_idim) ){
    printf("Filter: [%d %d], Image: [%d %d]\n", x_fdim, y_fdim, x_idim, y_idim);
    printf("FILTER dimensions larger than IMAGE dimensions.");
    exit(1);
    }*/

  if ( (x_step < 1) || ( y_step < 1) ){
    printf("STEP values must be greater than zero.\n");
    exit(1);
  }

  if ( (x_start < 0) || (x_start > x_idim) || 
       (y_start < 0) || (y_start > y_idim) ){
    printf("START values must lie between 1 and the image dimensions.\n");
    exit(1);
  }

  /*if ( (x_stop < x_start) || (x_stop > x_idim) ||
       (y_stop < y_start) || (y_stop > y_idim) ){
    printf("STOP values must lie between START and the image dimensions.");
    exit(1);
    }*/

  /* upConv has a bug for even-length kernels when using the reflect1, 
     extend, or repeat edge-handlers */
  // 

  printf("x_fdim=%d  y_fdim=%d  nd=%d dim0=%d dim1=%d\n", x_fdim, y_fdim, 
	 filt->nd, filt->dimensions[0], filt->dimensions[1]);
  for(x=0; x<x_fdim*y_fdim; x++)
    printf("filt[%d] = %f\n", x, (image_type)filt->data[x]);
  
  if ((!strcmp(edges,"reflect1") || !strcmp(edges,"extend") || 
       !strcmp(edges,"repeat"))
      &&
      ((x_fdim%2 == 0) || (y_fdim%2 == 0)))
    {
      dimensions2[0] = x_fdim * y_fdim;
      orig_filt = (PyArrayObject *)PyArray_FromDims(1, dimensions2, 
						    PyArray_DOUBLE);
      orig_filt->data = filt->data;
      /* orig_filt->data = malloc(x_fdim * y_fdim * sizeof(double));
	 for(x=0; x<x_fdim*y_fdim; x++)
	 orig_filt->data[x] = filt->data[x];
	 printf("checking orig_filt\n");
      for(x=0; x<x_fdim*y_fdim; x++)
      printf("orig_filt[%d] = %f\n", x, (image_type)orig_filt->data[x]); */
      orig_x = x_fdim; 
      orig_y = y_fdim;
      x_fdim = 2*(orig_x/2)+1;
      y_fdim = 2*(orig_y/2)+1;
      printf("changed fdim: x=%d y=%d\n", x_fdim, y_fdim);
      printf("going to malloc\n");
      dimensions3[0] = (x_fdim * y_fdim) + 1;
      filt = (PyArrayObject *)PyArray_FromDims(1, dimensions3, PyArray_DOUBLE);
      /* filt->data = malloc(x_fdim * y_fdim * sizeof(double));
	 filt->dimensions[0] = x_fdim * y_fdim;
	 filt->dimensions[1] = 1;
	 if (filt == NULL){
	 printf("Cannot allocate necessary temporary space");
	 exit(1);
	 } 
	 printf("done with malloc\n"); */

      // initialize all values
      /* for (x=0; x<x_fdim*y_fdim; x++)
	 filt->data[x] = (image_type)0.0; */

      // copy values back from orig_filt
      for (y=0; y<orig_y; y++){
	for (x=0; x<orig_x; x++){
	  printf("writing: %d   reading: %d\n", y*x_fdim+x, y*orig_x+x);
	  filt->data[y*x_fdim + x] = orig_filt->data[y*orig_x + x];
	  printf("%f  %f\n", (image_type)filt->data[y*x_fdim+x], 
		 (image_type)orig_filt->data[y*orig_x+x]);
	}
      }

    }

  //x_rdim = (x_stop-x_start+x_step-1) / x_step;
  //y_rdim = (y_stop-y_start+y_step-1) / y_step;
  x_rdim = x_stop; 
  y_rdim = y_stop;

  //dimensions[0] = (x_idim/y_step) * (y_idim/x_step);
  dimensions[0] = x_rdim * y_rdim;
  printf("dimensions[0]=%d  x_rdim=%d  y_rdim=%d\n", dimensions[0], x_rdim, 
	 y_rdim);
  if(arg3 == NULL){
    printf("flag 1\n");
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
  }else{
    printf("flag 2\n");
    result = (PyArrayObject *)PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE,
							   1, dimensions[0]);
  }
  
  printf("premalloc temp\n");
  double *temp = malloc(x_fdim * y_fdim * sizeof(double));
  if (temp == NULL){
    printf("Cannot allocate necessary temporary space");
    exit(1);
  }

  printf("i(%d, %d),f(%d, %d), r(%d, %d), X(%d, %d, %d), Y(%d, %d, %d), %s\n",
	 x_idim,y_idim,x_fdim,y_fdim,x_rdim,y_rdim,
	 x_start,x_step,x_stop,y_start,y_step,y_stop,edges);
  
  if (strcmp(edges,"circular") == 0)
    internal_wrap_expand((image_type *)image->data, (image_type *)filt->data,
			 x_fdim, y_fdim, x_start, x_step, x_stop, y_start, 
			 y_step, y_stop, (image_type *)result->data, x_rdim,
			 y_rdim);
  else
    internal_expand((image_type *)image->data, (image_type *)filt->data, 
		    (image_type *)temp, x_fdim, y_fdim, x_start, x_step, x_stop,
		    y_start, y_step, y_stop, (image_type *)result->data, x_stop,
		    y_stop, edges);
  
  free(temp);
  if(orig_x)
    free(filt);

  return PyArray_Return(result);
} 

static PyObject* py_pointOp(PyObject* self, PyObject* args)
{
  int x_idim, y_idim, lutsize;
  PyObject *poimage, *polut; 
  PyArrayObject *image, *lut;
  PyArrayObject *result = NULL;
  int warnings;
  double origin, increment;
  int dimensions[1];

  // parse input parameters
  if( !PyArg_ParseTuple(args, "iiOiOddi", &x_idim, &y_idim, &poimage, 
			&lutsize, &polut, &origin, &increment, &warnings) )
    return NULL;
  image = (PyArrayObject *)PyArray_ContiguousFromObject(poimage, PyArray_DOUBLE,
							1, x_idim * y_idim);
  if(image == NULL)
    return NULL;
  if(image->nd != 2 || image->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type double");
    return NULL;
  }

  lut = (PyArrayObject *)PyArray_ContiguousFromObject(polut, PyArray_DOUBLE,
						      1, lutsize);
  if(lut == NULL)
    return NULL;
  if(lut->nd != 1 || image->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be one-dimensional and of type double");
    return NULL;
  }

  dimensions[0] = x_idim * y_idim;
  result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
  
  internal_pointop(image->data, result->data, x_idim*y_idim, lut->data, 
		   lutsize, origin, increment, warnings);

  return PyArray_Return(result);

}

/*
 * Bind Python function names to our C functions
 */
static PyMethodDef c_methods[] = {
	{"corrDn", py_corrDn, METH_VARARGS},
	{"upConv", py_upConv, METH_VARARGS},
	{"pointOp", py_pointOp, METH_VARARGS},
	{NULL, NULL}
};

/*
 * Python calls this to let us initialize our module
 */
void initpyPyrCcode()
{
	(void) Py_InitModule("pyPyrCcode", c_methods);
	import_array();
}
