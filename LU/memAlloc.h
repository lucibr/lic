#include <stdlib.h>
#include <stdio.h>

void printVectorInt(int *v, int dim)
{
	int i;
	printf("\n");
	for(i = 0; i < dim; i++)
		printf("%d ", v[i]);
	printf("\n");
}

void printVectorDouble(double *v, int dim)
{
	int i;
	printf("\n");
	for(i = 0; i < dim; i++)
		printf("%.3lf ", v[i]);
	printf("\n");
}

void printMatrixInt(int **m, int nrl, int nrc) 
{
	int i, j;
	printf("\n");
	for(i = 0; i < nrl; i++)
	{
		for(j = 0; j < nrc; j++)
			printf( "%d ", m[i][j]);
		printf("\n");
	}
	printf("\n");
}

void printMatrixDouble(double **m, int nrl, int nrc) 
{
	int i, j;
	printf("\n");
	for(i = 0; i < nrl; i++)
	{
		for(j = 0; j < nrc; j++)
			printf( "%.3lf ", m[i][j]);
		printf("\n");
	}
	printf("\n");
}

int malloc2dint(int ***array, int nrl, int nrc)
{
	int i;
	/* allocate the n*m contiguous items */
	int *p = (int *)calloc(nrl*nrc, sizeof(int));
	if(!p)
		return -1;

	/* allocate the row pointers into the memory */
	(*array) = (int **)malloc(nrl*sizeof(int*));
	if(!array)
	{
		free(p);
		return -1;
	}

	/* set up the pointers into the contiguous memory */
	for (i=0; i<nrl; i++)
	       	(*array)[i] = &(p[i*nrc]);
	return 0;
}

int free2dint(int ***array)
{
	/* free the memory - the first element of the array is at the start */
	free(&((*array)[0][0]));

	/* free the pointers into the memory */
	free(*array);
	return 0;
}

int malloc2ddouble(double ***array, int nrl, int nrc)
{
	int i;
	/* allocate the n*m contiguous items */
	double *p = (double *)calloc(nrl*nrc, sizeof(double));
	if(!p)
		return -1;

	/* allocate the row pointers into the memory */
	(*array) = (double **)malloc(nrl*sizeof(double*));
	if(!array)
	{
		free(p);
		return -1;
	}

	/* set up the pointers into the contiguous memory */
	for (i=0; i<nrl; i++)
	       	(*array)[i] = &(p[i*nrc]);
	return 0;
}

int free2ddouble(double ***array)
{
	/* free the memory - the first element of the array is at the start */
	free(&((*array)[0][0]));

	/* free the pointers into the memory */
	free(*array);
	return 0;
}
