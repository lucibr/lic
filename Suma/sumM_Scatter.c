#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>
#include <time.h> 

void printVector(int *v, int dim)
{
	int i;
	printf("\n");
	for(i = 0; i < dim; i++)
		printf("%d ", v[i]);
	printf("\n");
}

//Tiparire matrice citita
void printMatrix(int **m, int nrl, int nrc) 
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

int main(int argc, char *argv[])
{
	int myID, nProcs, nrL, nrC, i, j;
	//Total number of matrix elements
	unsigned long n;
	//clock_t t1, t2;
	double start_time, stop_time;
	int rc, numTasks, rank, nrElem, nrSupElem, *toSend, *offsets, *receivedElems1, *receivedElems2, k;
	MPI_Status status;
	MPI_Request request;
	MPI_Datatype dataLayout_type, reducedDataLayout_type;
	FILE *in;
	int **mat1, **mat2;
	//Number of lines 
	nrL = atoi(argv[1]);
	//Number of columns 
	nrC = atoi(argv[2]);
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) 
	{ 
		printf ("\n\nError starting MPI program. Terminating.\n"); 
		MPI_Abort(MPI_COMM_WORLD, rc); 
	}
	//The parallel code begins here
	start_time = MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
	if(numTasks >= 2 && malloc2dint(&mat1, nrL, nrC) == 0 && malloc2dint(&mat2, nrL, nrC) == 0)
	{
		n = nrL * nrC;
		//Array will keep the number of elements to be sent to each process (for MPI_Scatterv)
		toSend = (int *) malloc(sizeof(int) * numTasks);
		//Array will keep the offsets relative to elements container address (for MPI_Scatterv)
		offsets = (int *) malloc(sizeof(int) * numTasks);
		//Store current process rank
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);

		nrElem = n/numTasks;
		nrSupElem = n%numTasks;
			
		//Memory allocation for the container of received elements
		receivedElems1 = (int *)malloc(sizeof(int) * (nrElem + 1));
		receivedElems2 = (int *)malloc(sizeof(int) * (nrElem + 1));

		//Process of rank 0 - first process (the  master process)
		if(rank == 0)
		{
			printf("\nNumber of available processes: %d\n", numTasks);
			printf("\nProcess %d: Reading data from file...\n", rank);
			
			in = fopen("34.txt", "r");
			for(i = 0; i < nrL; i++)
			{
				for(j = 0; j < nrC; j++)
				{
					k = fscanf(in, "%d",&mat1[i][j]);
					mat2[i][j] = mat1[i][j];
				}
			}
			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			printf("\nProcess %d: File closed...\n", rank);			
			printMatrix(mat1, nrL, nrC);
			printMatrix(mat2, nrL, nrC);
		}
		//Defining offsets and number of elements to be sent to each process (information is available in every process)
		for(i = 0; i < numTasks; i++)
		{
			//For a distribution as balanced as possible, for the first (TotalElemNumber%TasksNumber) processes a supplementary elem is considered
			//Normaly, each process receives (TotalElemNumber/TasksNumber) elements
			if(i < nrSupElem)
			{
				//Send nrElem + 1
				toSend[i] = nrElem + 1;
			}
			else
			{
				//Send nrElem
				toSend[i] = nrElem;
			}
			//For rank o process elements are taken starting with index (offset) 0
			if(i == 0)
			{
				offsets[0] = 0;
			}
			else if(i < nrSupElem )
			{
				//offsets[i] = (i*nrElem+nrSupElem-i);
				offsets[i] = 0;
			}
			else
			{
				offsets[i] = (i*nrElem+nrSupElem);
			}
		}
		//printf("\nProcess %d: Scattering...\n", rank);
		MPI_Scatterv(&(mat1[0][0]), toSend, offsets, MPI_INT, receivedElems1, nrElem + 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(&(mat2[0][0]), toSend, offsets, MPI_INT, receivedElems2, nrElem + 1, MPI_INT, 0, MPI_COMM_WORLD);
		//printf("\nProcess %d: Scattering succeeded!\n", rank);

		
		for(i = 0; i < toSend[rank]; i++)
		{
			receivedElems1[i] += receivedElems2[i];
		}
		//printf("\nProcess %d - \nLocal operations finished...\n", rank);
		MPI_Gatherv(receivedElems1, toSend[rank], MPI_INT, &(mat1[0][0]), toSend, offsets, MPI_INT, 0, MPI_COMM_WORLD);
		if(rank == 0)
		{
			printf("\nThe sum is:\n");
			printMatrix(mat1, nrL, nrC);
			free2dint(&mat1);
			free2dint(&mat2);
			stop_time = MPI_Wtime();
			printf("\nRuntime is: %f\n", stop_time - start_time);
		}
		free(toSend);
		free(offsets);
		free(receivedElems1);
		free(receivedElems2);

	}
	else
	{
		printf("\nNumber of available processes: %d\n", numTasks);
		printf("\nInsufficient resources (processors/memory).");
	}
	MPI_Finalize();
	return 0;	
}
