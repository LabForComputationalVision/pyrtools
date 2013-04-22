#include <Python.h>
#include "Numeric/arrayobject.h"
#include "convolve.h"

/* compile with: gcc -shared -I/usr/include/python2.6/ -lpython2.6 -o myModule.so -fPIC myModule.c convolve.c convolve.h edges.c wrap.c*/

#define notDblMtx(it) (!mxIsNumeric(it) || !mxIsDouble(it) || mxIsSparse(it) || mxIsComplex(it))


static PyObject* py_corrDn(PyObject* self, PyObject* args)
//void mexFunction(int nlhs,	     /* Num return vals on lhs */
//		 mxArray *plhs[],    /* Matrices on lhs      */
//		 int nrhs,	     /* Num args on rhs    */
//		 const mxArray *prhs[]     /* Matrices on rhs */
//		 )
{
  //double *image,*filt, *temp, *result;
  int x_fdim, y_fdim;
  int Nargs;
  int x_idim, y_idim;
  //int x_rdim, y_rdim;
  //int x_start = 1;
  //int x_step = 1;
  //int y_start = 1;
  //int y_step = 1;
  //int x_stop, y_stop;
  //mxArray *arg;
  PyObject *arg1, *arg2;
  PyArrayObject *image, *filt, *result;
  /*double *mxMat;  */
  char *edges;
  int x_step, y_step, x_start, y_start, x_stop, y_stop, x_rdim, y_rdim;
  int dimensions[1];
  int i;
  double *ix;
  int *ixd, *rxd;
  double *fx;
  image_type *rx;

  // parse input parameters
  // FIX: make some parameters optional
  // FIX: can we set the format so we don't need ix & fx?
  if( !PyArg_ParseTuple(args, "iiOiiOsiiiiii", &x_idim, &y_idim, &arg1, 
			&x_fdim, &y_fdim, &arg2, &edges, &x_step, &y_step, 
			&x_start, &y_start, &x_stop, &y_stop) )
    return NULL;
  image = (PyArrayObject *)PyArray_ContiguousFromObject(arg1, PyArray_INT, 1, 
							x_idim * y_idim);
  printf("flag 1\n");
  ixd = (int *)image->data;
  ix = malloc(x_idim * y_idim * sizeof(double));
  for(i=0; i<x_idim*y_idim; i++){
    ix[i] = (double)ixd[i];
    printf("ix[%d]=%f\n",i,ix[i]);
  }
  printf("flag 2\n");

  if(image == NULL)
    return NULL;
  if(image->nd != 2 || image->descr->type_num != PyArray_INT){
    PyErr_SetString(PyExc_ValueError, 
		    "array must be two-dimensional and of type int");
    return NULL;
  }

  filt = (PyArrayObject *)PyArray_ContiguousFromObject(arg2, PyArray_DOUBLE, 
						       1, x_fdim * y_fdim);
  fx = (double *)filt->data;

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


  dimensions[0] = x_idim * y_idim;
  result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_INT);
  double *temp = malloc(x_fdim * y_fdim * sizeof(double));
  if (temp == NULL){
    printf("Cannot allocate necessary temporary space");
    exit(1);
  }

  /*
  printf("i(%d, %d), f(%d, %d), r(%d, %d), X(%d, %d, %d), Y(%d, %d, %d), %s\n",
	 x_idim,y_idim,x_fdim,y_fdim,x_rdim,y_rdim,
	 x_start,x_step,x_stop,y_start,y_step,y_stop,edges); */

  rx = malloc(x_idim * y_idim * sizeof(image_type));
  if (strcmp(edges,"circular") == 0)
    internal_wrap_reduce((image_type *)ix, x_idim, y_idim, 
			 (image_type *)fx, x_fdim, y_fdim,
  			 x_start, x_step, x_stop, y_start, y_step, y_stop,
  			 (image_type *)rx);
  else 
    internal_reduce((image_type *)ix, x_idim, y_idim, 
		    (image_type *)fx, temp, x_fdim, y_fdim,
		    x_start, x_step, x_stop, y_start, y_step, y_stop,
		    (image_type *)rx, edges);
  

  //for(i=500; i<510; i++)
  //printf("%d: %f\n", i, rx[i]);
  //printf("%d: %d\n", i, ix[i]);
  rxd = malloc(x_idim * y_idim * sizeof(int));
  for(i=0; i<x_idim*y_idim; i++){
    rxd[i] = (int)floor(rx[i]);
    if(rxd[i] > 0)
      printf("rxd[%d]=%d\n",i,rxd[i]);
  }
  
  result->data = rxd;
  free(temp);
  free(rx);
  return PyArray_Return(result);

} 



/*
 * Another function to be called from Python
 */
static PyObject* py_myOtherFunction(PyObject* self, PyObject* args)
{
	double x, y;
	PyArg_ParseTuple(args, "dd", &x, &y);
	return Py_BuildValue("d", x*y);
}

/*
 * Bind Python function names to our C functions
 */
static PyMethodDef myModule_methods[] = {
	{"corrDn", py_corrDn, METH_VARARGS},
	{"myOtherFunction", py_myOtherFunction, METH_VARARGS},
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
