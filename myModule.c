#include <Python.h>
#include "Numeric/arrayobject.h"
#include "convolve.h"

/* compile with: gcc -shared -I/usr/include/python2.6/ -lpython2.6 -o myModule.so -fPIC myModule.c convolve.c convolve.h edges.c wrap.c*/

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
    printf("FILTER dimensions larger than IMAGE dimensions.");
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


  /*printf("i(%d, %d), f(%d, %d), r(%d, %d), X(%d, %d, %d), Y(%d, %d, %d),%s\n",
    x_idim,y_idim,x_fdim,y_fdim,x_rdim,y_rdim,
    x_start,x_step,x_stop,y_start,y_step,y_stop,edges); */

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
  PyArrayObject *image, *filt, *orig_filt, *result;
  int orig_x = 0;
  int orig_y, x, y;
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
		    "array must be two-dimensional and of type double");
    return NULL;
  }

  filt = (PyArrayObject *)PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 
						       1, x_fdim * y_fdim);
  if(filt == NULL)
    return NULL;
  if(filt->nd != 2 || filt->descr->type_num != PyArray_DOUBLE){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type double");
    return NULL;
  }

  /*if( (x_fdim > x_idim) || (y_fdim > y_idim) ){
    printf("Filter: [%d %d], Image: [%d %d]\n", x_fdim, y_fdim, x_idim, y_idim);
    printf("FILTER dimensions larger than IMAGE dimensions.");
    exit(1);
    }*/

  if ( (x_step < 1) || ( y_step < 1) ){
    printf("STEP values must be greater than zero.");
    exit(1);
  }

  if ( (x_start < 0) || (x_start > x_idim) || 
       (y_start < 0) || (y_start > y_idim) ){
    printf("START values must lie between 1 and the image dimensions.");
    exit(1);
  }

  /*if ( (x_stop < x_start) || (x_stop > x_idim) ||
       (y_stop < y_start) || (y_stop > y_idim) ){
    printf("STOP values must lie between START and the image dimensions.");
    exit(1);
    }*/

  /* upConv has a bug for even-length kernels when using the reflect1, 
     extend, or repeat edge-handlers */
  if ((!strcmp(edges,"reflect1") || !strcmp(edges,"extend") || 
       !strcmp(edges,"repeat"))
      &&
      ((x_fdim%2 == 0) || (y_fdim%2 == 0)))
    {
      orig_filt = filt;
      orig_x = x_fdim; 
      orig_y = y_fdim;
      x_fdim = 2*(orig_x/2)+1;
      y_fdim = 2*(orig_y/2)+1;
      filt = malloc(x_fdim * y_fdim * sizeof(double));
      if (filt == NULL){
	printf("Cannot allocate necessary temporary space");
	exit(1);
      }
      for (y=0; y<orig_y; y++)
	for (x=0; x<orig_x; x++)
	    filt[y*x_fdim + x] = orig_filt[y*orig_x + x];
    }

  //x_rdim = (x_stop-x_start+x_step-1) / x_step;
  //y_rdim = (y_stop-y_start+y_step-1) / y_step;
  x_rdim = x_stop; 
  y_rdim = y_stop;

  //dimensions[0] = (x_idim/y_step) * (y_idim/x_step);
  dimensions[0] = x_rdim * y_rdim;
  if(arg3 == NULL)
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
  else
    result = (PyArrayObject *)PyArray_ContiguousFromObject(arg3, PyArray_DOUBLE,
							   1, dimensions[0]);
  
  double *temp = malloc(x_fdim * y_fdim * sizeof(double));
  if (temp == NULL){
    printf("Cannot allocate necessary temporary space");
    exit(1);
  }

  /*printf("i(%d, %d),f(%d, %d), r(%d, %d), X(%d, %d, %d), Y(%d, %d, %d), %s\n",
    x_idim,y_idim,x_fdim,y_fdim,x_rdim,y_rdim,
    x_start,x_step,x_stop,y_start,y_step,y_stop,edges); */
  
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

/*
 * Bind Python function names to our C functions
 */
static PyMethodDef myModule_methods[] = {
	{"corrDn", py_corrDn, METH_VARARGS},
	{"upConv", py_upConv, METH_VARARGS},
	{NULL, NULL}
};

/*
 * Python calls this to let us initialize our module
 */
void initmyModule()
{
	(void) Py_InitModule("myModule", myModule_methods);
	import_array();
}
