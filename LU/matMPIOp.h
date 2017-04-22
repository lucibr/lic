#ifndef MATMPIOP_H
#define MATMPIOP_H

//void MPI_Framework_Init(int, char*, int*)
//void MPI_Framework_Stop();
int malloc2ddouble(double ***, int, int);
void printErrorMessage(int, int, char*);
void printMatrixDouble(double **, int, int);
void printVectorDouble(double *, int);
int prodMV_DD(double** , int, int, double *, int, double **, double *, int);

#endif
