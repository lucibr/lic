#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>
#include <time.h> 
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

void printMatrixDouble(double **m, int nrl, int nrc) 
{
	int i, j;
	printf("\n");
	for(i = 0; i < nrl; i++)
	{
		for(j = 0; j < nrc; j++)
			printf( "%.2lf ", m[i][j]);
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

int decompLU(double **mat, double **L, double **U, int **P, int nrL, int nrC)
{
	if(nrL != nrC)
	{
		return -2;
	}
	//Varianta neparalelizata LU = PA
	int i, j, k;
	int lineMaxPivot;
	double maxPivot, aux;
	for(i = 0; i < nrL - 1; i++)
	{
		//Get the max value on the 'i' column (pivot value)
		lineMaxPivot = i;
		maxPivot = fabs(U[i][i]);
		//printf("\nInitial pivot %f", maxPivot);
		for(j = i+1; j < nrL; j++)
		{
			if(abs(U[j][i]) > maxPivot)
			{
				maxPivot = fabs(U[j][i]);
				lineMaxPivot = j;
			}
			
		}
		if(fabs(maxPivot) <= EPS)
		{
			//pivot value to small
			return -1;
		}
		if(i != lineMaxPivot)
		{
			//printf("\nInterschimbare linii %d si %d", i, lineMaxPivot);
			//The pivot value is not on the current line --> interchange the lines
			//Interchange lines in the original matrix
			for(j = 0; j < nrC; j++)
			{
				aux = U[i][j];
				U[i][j] = U[lineMaxPivot][j];
				U[lineMaxPivot][j] = aux;

				aux = L[i][j];
				L[i][j] = L[lineMaxPivot][j];
				L[lineMaxPivot][j] = aux; 
			}
			//Interchange lines in P matrix
			P[i][i] = 0;
			P[lineMaxPivot][lineMaxPivot] = 0;
			P[i][lineMaxPivot] = 1;
			P[lineMaxPivot][i] = 1;


			//Interchange lines in L?? TO CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		}
		//printf("\nValoare pivot %f", maxPivot);
		// L coefficients (matricea multiplicatorilor)
		L[i][i] = 1;
		for(j = i+1; j < nrL; j++)
		{
			L[i][j] = 0; 
			L[j][i] = U[j][i]/maxPivot;
		}
		// U coefficients (matricea finala)
		for(j = i+1; j <nrL; j++)
			for(k = 0; k < nrC; k++)
			{
				//printf("\nL[%d][%d]=%lf", j, i, L[j][i]);
				U[j][k] = U[j][k] - L[j][i] * U[i][k]; 
			}

		//printf("\nDupa pasul %d:\n", i + 1);
		//printf("\nMatricea L:\n");
		//printMatrixDouble(L, nrL, nrC);
		//printf("\nMatricea U:\n");
		//printMatrixDouble(U, nrL, nrC);
		//printf("\nMatricea P:\n");
		//printMatrixInt(P, nrL, nrC);
	}
	return 0;
}

int parallelDecompLU(double **mat, double **L, double **U, int **P, int nrR, int nrC, int nProcs, int PC, int PR)
{

	if(nrR != nrC)
	{
		//Not square matrix
		return -2;
	}
	if(PC * PR > nProcs)
	{
		//Not sufficient resources for the specified processor grid
		return -3;
	}
	if(PC != PR)
	{
		//For square matrix the processes grid should also be square
		return -4;	
	}
	if(nrR % PR != 0 || nrC % PC != 0)
	{
		//Grid cannot be matched with the matrix size
		return -5;
	}
	int rBD, cBD, **processes;	//row & column block sizes
	rBD = nrR/PR;
	cBD = nrC/PC;
	if(rBD < 2 || cBD < 2)
	{
		//Parallel computation not needed (less than 2x2 matrix blocks allocated to each processor)
		printf("\nParallel computation not needed (less than 2x2 matrix blocks allocated to each processor).\n");
		return decompLU(mat, L, U, P, nrR, nrC);
	}
	else if(malloc2dint(&processes, PR, PC) == 0)
	{
		int i, j, k;
		k = 0;
		//Create the processes matrix
		for(i = 0; i < PR; i++)
			for(j = 0; j < PC; j++)
			{
				processes[i][j] = k++;
			}


		printMatrixInt(processes, PR, PC);
		free2dint(&processes);
		return 0;
	}
	else
	{
		printf("\nError in memory allocation...");
		return -5;
	}

}

int main(int argc, char *argv[])
{
	int myID, nProcs, nrL, nrC, i, j, k;
	int rc, numTasks, rank, res, PR, PC;
	FILE *in;
	double **mat, **U, **L;
	int **P;
	//Number of matrix lines
	nrL = atoi(argv[1]);
	//Number of matrix columns
	nrC = atoi(argv[2]);
	//Number of process grid rows
	PR = atoi(argv[3]);
	//Number of process grid columns
	PC = atoi(argv[4]);
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) 
	{ 
		printf ("\n\nError starting MPI program. Terminating.\n"); 
		MPI_Abort(MPI_COMM_WORLD, rc); 
	}
	MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(malloc2ddouble(&mat, nrL, nrC) == 0 && malloc2dint(&P, nrL, nrC) == 0 && malloc2ddouble(&L, nrL, nrC) == 0 && malloc2ddouble(&U, nrL, nrC) == 0)
	{
		//if(rank == 0)
		//{
			in = fopen("34.txt", "r");
			for(i = 0; i < nrL; i++)
			{
				for(j = 0; j < nrC; j++)
				{
					k = fscanf(in, "%lf",&mat[i][j]);
					if(i == j)
					{
						P[i][j] = 1;
					}
					else
					{
						P[i][j] = 0;
					}
					U[i][j] = mat[i][j];
					L[i][j] = 0;
				}
			}
			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			printf("\nProcess %d: File closed...\n", rank);			
			//printMatrixDouble(mat, nrL, nrC);
		//}

		//res = decompLU(mat, L, U, P, nrL, nrC);
		res = parallelDecompLU(mat, L, U, P, nrL, nrC, numTasks, PR, PC);
		if(res == -1)
		{
			printf("\nPivot value too small!\n");
		}
		else if(res == -2)
		{
			printf("\nMatrix is not square!\n");
		}
		else if(res == -3)
		{
			printf("\nNot sufficient resources for the specified processor grid...\n");
		}
		else if(res == -4)
		{
			printf("\nFor square matrix the processes grid should also be square...");
		}
		else if(res == -5)
		{
			printf("\n!!!Error in memory allocation...");
		}
		else
		{
			//printf("\nMatricea L:\n");
			//printMatrixDouble(L, nrL, nrC);
			//printf("\nMatricea U:\n");
			//printMatrixDouble(U, nrL, nrC);
			//printf("\nMatricea P:\n");
			//printMatrixInt(P, nrL, nrC);
		}

		free2ddouble(&mat);
		free2dint(&P);
		free2ddouble(&L);
		free2ddouble(&U);
	}
	else
	{
		printf("\nError in memory allocation...");
	}
	MPI_Finalize();
	return 0;	
}
