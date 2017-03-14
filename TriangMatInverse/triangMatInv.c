#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <string.h>
#include <math.h>

#define EPS 0.01


void printVector(int *v, int dim)
{
	int i;
	printf("\n");
	for(i = 0; i < dim; i++)
		printf("%d ", v[i]);
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

int malloc2dint(int ***array, int nrl, int nrc)
{
	int i;
	/* allocate the n*m contiguous items */
	int *p = (int *)malloc(nrl*nrc*sizeof(int));
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

void printMatrixDouble(double **m, int nrl, int nrc) 
{
	int i, j;
	printf("\n");
	for(i = 0; i < nrl; i++)
	{
		for(j = 0; j < nrc; j++)
			printf( "%.4lf ", m[i][j]);
		printf("\n");
	}
	printf("\n");
}

int malloc2ddouble(double ***array, int nrl, int nrc)
{
	int i;
	/* allocate the n*m contiguous items */
	double *p = (double *)malloc(nrl*nrc*sizeof(double));
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

int infMatInv(double **mat, double **inv, int dim)
{
	int i, j, k;
	double multiplier;
	//Initialize I matrix
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if(i == j)
			{
				inv[i][j] = 1;
			}
			else
			{
				inv[i][j] = 0;
			}
		}
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j <= i; j++)
		{
			if(i == j)
			{
				multiplier = mat[i][i];
				for(k = 0; k <= i; k++)
				{
					mat[i][k] /= multiplier;
					inv[i][k] /= multiplier;
				}
			}
			else
			{
				multiplier = mat[i][j];
				for(k = 0; k < i; k++)
				{
					mat[i][k] = mat[i][k] - multiplier * mat[j][k];
					inv[i][k] = inv[i][k] - multiplier * inv[j][k];
				}
			}
		}
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if((i == j && fabs(mat[i][j]-1) > EPS) || (i != j && fabs(mat[i][j]) > EPS))
			{
				return -2;
			}
		}
	}
	return 0;
}


int supMatInv(double **mat, double **inv, int dim)
{
	int i, j, k;
	double multiplier;
	//Initialize I matrix
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if(i == j)
			{
				inv[i][j] = 1;
			}
			else
			{
				inv[i][j] = 0;
			}
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = dim - 1; j >= i; j--)
		{
			if(i == j)
			{
				multiplier = mat[i][i];
				for(k = dim - 1; k >= i; k--)
				{
					mat[i][k] /= multiplier;
					inv[i][k] /= multiplier;
				}
			}
			else
			{
				multiplier = mat[i][j];
				for(k = dim - 1; k > i; k--)
				{
					mat[i][k] = mat[i][k] - multiplier * mat[j][k];
					inv[i][k] = inv[i][k] - multiplier * inv[j][k];
				}
			}
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = dim - 1; j >= 0; j--)
		{
			if((i == j && fabs(mat[i][j]-1) > EPS) || (i != j && fabs(mat[i][j]) > EPS))
			{
				return -2;
			}
		}
	}
	return 0;
}



int main(int argc, char *argv[])
{
	int dim, i, j, k;
	FILE *in;
	double **mat, **inv;
	//Matrix dimension
	dim = atoi(argv[1]);
	if(malloc2ddouble(&mat, dim, dim) == 0 && malloc2ddouble(&inv, dim, dim) == 0)
	{
		in = fopen("3L2.txt", "r");
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				k = fscanf(in, "%lf",&mat[i][j]);
				if(k);
			}
		}
		printf("\nReading done...");
		fclose(in);
		printf("\nFile closed...");			
		printMatrixDouble(mat, dim, dim);
		int result = infMatInv(mat, inv, dim);
		if(result == -1)
		{
			printf("\n\nInversion ERROR: Memory allocation failed...\n");
		}
		else if(result == -2)
		{
			printf("\n\nInversion ERROR: Matrix cannot be inverted...\n");
		}
		else
		{	
			printf("\nThe matrix inverse is:\n");
			printMatrixDouble(inv, dim, dim);
		}
		free2ddouble(&mat);
		free2ddouble(&inv);
	}
	else
	{
		printf("\nInsufficient resources (processors/memory).");
	}
	return 0;	
}
