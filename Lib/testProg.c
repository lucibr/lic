#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "mpi.h"
#include "matMPIOp.h"

int main(int argc, char *argv[])
{
	int nrL1, nrC1, nrL2, nrC2, i, j, k;
	int rc, numTasks, rank;
	FILE *in;
	int **mat1, **mat2;
	//Number of lines of first matrix
	nrL1 = atoi(argv[1]);
	//Number of columns of first matrix
	nrC1 = atoi(argv[2]);
	//Number of lines of first matrix
	nrL2 = atoi(argv[3]);
	//Number of columns of first matrix
	nrC2 = atoi(argv[4]);

	char *error;
	void *handle;

	handle = dlopen("libmatMPIOp.so", RTLD_LAZY);
	if(!handle)
	{
		fputs(dlerror(), stderr);
		exit(1);
	}

	prodM = dlsym(handle, "prodM");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	sumM = dlsym(handle, "sumM");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	malloc2dint = dlsym(handle, "malloc2dint");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	free2dint = dlsym(handle, "free2dint");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	printMatrix = dlsym(handle, "printMatrix");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}
		
	rc = MPI_Init(&argc, &argv);

	if (rc != MPI_SUCCESS) 
	{ 
		printf ("\n\nError starting MPI program. Terminating.\n"); 
		MPI_Abort(MPI_COMM_WORLD, rc); 
	}
	MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
	if(malloc2dint(&mat1, nrL1, nrC1) == 0 && malloc2dint(&mat2, nrL2, nrC2) == 0)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		//Process of rank 0 - first process (the  master process)
		if(rank == 0)
		{
			printf("\nNumber of available processes: %d\n", numTasks);
			printf("\nProcess %d: Reading data from file...\n", rank);
			
			in = fopen("numere.txt", "r");
			for(i = 0; i < nrL1; i++)
			{
				for(j = 0; j < nrC1; j++)
				{
					k = fscanf(in, "%d",&mat1[i][j]);
				}
			}
			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			printf("\nProcess %d: File closed...\n", rank);		
			//printMatrix(mat1, nrL1, nrC1);
	
			in = fopen("numere.txt", "r");
			for(i = 0; i < nrL2; i++)
			{
				for(j = 0; j < nrC2; j++)
				{
					k = fscanf(in, "%d",&mat2[i][j]);
				}
			}
			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			printf("\nProcess %d: File closed...\n", rank);
			//printMatrix(mat2, nrL2, nrC2);
		}
		int **result = prodM(mat1, nrL1, nrC1 ,mat2, nrL2, nrC2, numTasks);
		if(rank == 0)
		{
			if( result == NULL)
			{
				printf("\n\nProcess %d: ERROR: Product failed!\n", rank);
			}
			else
			{	
				printf("\nThe product is:\n");
				//printMatrix(result, nrL1, nrC2);
			}
		}
		
		result = sumM(mat1, mat2, nrL1, nrC1, numTasks);
		if(rank == 0)
		{
			if( result == NULL)
			{
				printf("\n\nProcess %d: ERROR: Sum failed!\n", rank);
			}
			else
			{	
				printf("\nThe sum is:\n");
				//printMatrix(result, nrL1, nrC2);
			}
			free2dint(&mat1);
			free2dint(&mat2);
		}
	}
	else
	{
		printf("\nInsufficient resources (processors/memory).");
	}
	MPI_Finalize();

	dlclose(handle);
	return 0;	
}
