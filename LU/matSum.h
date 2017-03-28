#include "memAlloc.h"
#include "MPIWrapper.h"


int sumM_DD(double **mat1, double **mat2, double f1, double f2, int nrL, int nrC, double *runtime, double ***result)
{
	int i, j;
	double start_time, stop_time;
	start_time = MPI_Wtime();
	for(i = 0; i < nrL; i++)
	{
		for(j = 0; j < nrC; j++)
		{
			(*result)[i][j] = mat1[i][j] * f1 + mat2[i][j] * f2;
		}
	}
	stop_time = MPI_Wtime();
	*runtime = stop_time - start_time;
	return 0;
}

int sumM_DD_P(double **mat1, double **mat2, double f1, double f2, int nrL, int nrC, int nProcs, double *runtime, double ***result)
{
	int i, rankL, nrElem, nrSupElem, *toSend, *offsets;
	double start_time, stop_time, *receivedElems1, *receivedElems2, factor1 = f1, factor2 = f2;
	//Total number of matrix elements
	unsigned long n;

	if(nProcs == 1)
	{
		return sumM_DD(mat1, mat2, f1, f2, nrL, nrC, runtime, result);
	}

	n = nrL * nrC;
	//Array will keep the number of elements to be sent to each process (for MPI_Scatterv)

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	//The parallel code begins here
	start_time = MPI_Wtime();

	//Array will keep the number of elements to be sent to each process (for MPI_Scatterv)
	toSend = (int *) malloc(sizeof(int) * nProcs);
	//Array will keep the offsets relative to elements container address (for MPI_Scatterv)
	offsets = (int *) malloc(sizeof(int) * nProcs);

	nrElem = n / nProcs;	
	nrSupElem = n % nProcs;

	//Memory allocation for the container of received elements
	receivedElems1 = (double *)malloc(sizeof(double) * (nrElem + 1));
	receivedElems2 = (double *)malloc(sizeof(double) * (nrElem + 1));
					
	//Defining offsets and number of elements to be sent to each process (information is available in every process)	
	for(i = 0; i < nProcs; i++)
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
			offsets[i] = (i*nrElem+nrSupElem-i);
		}
		else
		{
			offsets[i] = (i*nrElem+nrSupElem);
		}
	}
	//printf("\nProcess %d: Scattering...\n", rank);
	MPI_Bcast(&factor1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&factor2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(&(mat1[0][0]), toSend, offsets, MPI_DOUBLE, receivedElems1, nrElem + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(&(mat2[0][0]), toSend, offsets, MPI_DOUBLE, receivedElems2, nrElem + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//printf("\nProcess %d: Scattering succeeded!\n", rankL);
	//printf("\nProcess %d - received: ", rankL);
	//printVector(receivedElems1, toSend[rankL]);
	for(i = 0; i < toSend[rankL]; i++)
	{
		receivedElems1[i] = receivedElems1[i] * factor1 + receivedElems2[i] * factor2;
	}
	//printf("\nProcess %d - \nLocal operations finished...\n", rank);
	if(rankL == 0)
	{
		if(malloc2ddouble(result, nrL, nrC) != 0)
		{	
			return -5;
		}
	}
	MPI_Gatherv(receivedElems1, toSend[rankL], MPI_INT, &((*result)[0][0]), toSend, offsets, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rankL == 0)
	{
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
	}
	
	free(toSend);
	free(offsets);
	free(receivedElems1);
	free(receivedElems2);		
	return 0;
}
