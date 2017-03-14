#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h> 

void printVector(int *v, int dim)
{
	int i;
	printf("\n");
	for(i = 0; i < dim; i++)
		printf("%d ", v[i]);
	printf("\n");
}

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

int** sumM(int **mat1, int **mat2, int nrL, int nrC, int nProcs)
{
	int numTasksL, *ranks, i, j, rankL, nrElem, nrSupElem, *toSend, *offsets, *receivedElems1, *receivedElems2;
	//clock_t t1, t2;
	double start_time, stop_time;
	//Total number of matrix elements
	unsigned long n;

	int testRes = -100;

	n = nrL * nrC;
	MPI_Group resizedGroup, group_world;
	MPI_Comm  resizedCommunicator;
	MPI_Comm_size(MPI_COMM_WORLD,&numTasksL);
	if(numTasksL < nProcs)
	{
		//Insufficient available processes
		return NULL;
	}
	else if(nProcs > n)
	{
		if(rankL == 0)
			printf("\nToo many processes specified: %lu elements have to be computed by %d processes. Aborting...\n", n, nProcs); 
		return NULL;
	}
	else if(nProcs == 1)
	{
		start_time = MPI_Wtime();
		for(i = 0; i < nrL; i++)
			for(j = 0; j < nrC; j++)
				mat1[i][j] += mat2[i][j];
		stop_time = MPI_Wtime();
		printf("\nRuntime is: %f\n", stop_time - start_time);
		return mat1;
	}
	else
	{
		numTasksL = nProcs;
		//Array will keep the number of elements to be sent to each process (for MPI_Scatterv)
		ranks = (int *) malloc(sizeof(int) * numTasksL);
		for(i = 0; i < numTasksL; i++)
		{
			ranks[i] = i;
		}
		MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
		if((testRes = MPI_Comm_group(MPI_COMM_WORLD, &group_world)) == MPI_SUCCESS)
		{
			//printf("\nGroup_world test result: %d\n",testRes);
			//Defining a group with nProcs processes
			if((testRes = MPI_Group_incl(group_world, numTasksL, ranks, &resizedGroup)) == MPI_SUCCESS)
			{
				//printf("\nMPI_Group_incl test result: %d\n",testRes);
				if((testRes = MPI_Comm_create(MPI_COMM_WORLD, resizedGroup, &resizedCommunicator)) == MPI_SUCCESS && rankL < nProcs)
				{
					//printf("\nMPI_Comm_create test result: %d\n",testRes);
					//The parallel code begins here
					start_time = MPI_Wtime();
					MPI_Comm_rank(resizedCommunicator, &rankL);
					//Array will keep the number of elements to be sent to each process (for MPI_Scatterv)
					toSend = (int *) malloc(sizeof(int) * numTasksL);
					//Array will keep the offsets relative to elements container address (for MPI_Scatterv)
					offsets = (int *) malloc(sizeof(int) * numTasksL);
	
					nrElem = n/numTasksL;	
					nrSupElem = n%numTasksL;
				
					//Memory allocation for the container of received elements
					receivedElems1 = (int *)malloc(sizeof(int) * (nrElem + 1));
					receivedElems2 = (int *)malloc(sizeof(int) * (nrElem + 1));
					
					//Defining offsets and number of elements to be sent to each process (information is available in every process)	
					for(i = 0; i < numTasksL; i++)
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
					MPI_Scatterv(&(mat1[0][0]), toSend, offsets, MPI_INT, receivedElems1, nrElem + 1, MPI_INT, 0, resizedCommunicator);
					MPI_Scatterv(&(mat2[0][0]), toSend, offsets, MPI_INT, receivedElems2, nrElem + 1, MPI_INT, 0, resizedCommunicator);
					//printf("\nProcess %d: Scattering succeeded!\n", rankL);
					//printf("\nProcess %d - received: ", rankL);
					//printVector(receivedElems1, toSend[rankL]);
					for(i = 0; i < toSend[rankL]; i++)
					{
						receivedElems1[i] += receivedElems2[i];
					}
					//printf("\nProcess %d - \nLocal operations finished...\n", rank);
					MPI_Gatherv(receivedElems1, toSend[rankL], MPI_INT, &(mat1[0][0]), toSend, offsets, MPI_INT, 0, 	resizedCommunicator);
					
					if(rankL == 0)
					{
						stop_time = MPI_Wtime();
						printf("\nRuntime is: %f\n", stop_time - start_time);
					}
					
					free(toSend);
					free(offsets);
					free(receivedElems1);
					free(receivedElems2);
					free(ranks);					
					if (rankL == 0)
					{
						MPI_Comm_free(&resizedCommunicator);
						MPI_Group_free(&resizedGroup);
						MPI_Group_free(&group_world);
					}
					return mat1;
				}
				else
				{
					free(ranks);
					if(rankL == 0)
					{
						MPI_Group_free(&resizedGroup);
						MPI_Group_free(&group_world);
					}
					return NULL;
				}
			}
			else
			{
				free(ranks);
				if(rankL == 0)
				{
					MPI_Group_free(&group_world);
				}
				return NULL;
			}
		}
		else
		{
			free(ranks);
			return NULL;
		}
	}
}

int** prodM(int **mat1, int nrL1, int nrC1, int **mat2, int nrL2, int nrC2, int nProcs)
{
	//The Output Data decomposition method is applied
	int numTasksL, *ranks, i, j, k, rankL, nrElem, nrSupElem, maxNumberOfLines, 
		*toGet, //will store the number of elements from the result matrix to be obtained from each process
		*offsets, //will store the offsets where the elements from each process will be storred (offsets of **resultMat)
            	*receivedLineElements, //Will store the first matrix elements sent to each process
		*receivedLineOffsets, //will store the first matrix start offsets of lines to be sent to each process
		*recevedLinesNumbers, //Will store the numbers of lines from the first matrix to be sent to each process
		**resultMat, //will store the product AxB result
		*resultElem; //will store the elements computed by each process
	//clock_t t1, t2;
	double start_time, stop_time;
	//Total number of matrix elements
	unsigned long n;
	int testRes = -100;

	//Number of elements in the result matrix
	n = nrL1 * nrC2;
	MPI_Group resizedGroup, group_world;
	MPI_Comm  resizedCommunicator;
	MPI_Comm_size(MPI_COMM_WORLD,&numTasksL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	if(numTasksL < nProcs)
	{
		if(rankL == 0)
			printf("\nInsufficient resources: %d processes requested - only %d are available\n", nProcs, numTasksL);
		//Insufficient available processes
		return NULL;
	}
	else if(nProcs > n)
	{
		if(rankL == 0)
			printf("\nToo many processes specified: %lu elements have to be computed by %d processes. Aborting...\n", n, nProcs); 
		return NULL;
	}
	else if(nrC1 != nrL2)
	{
		if(rankL == 0)
			printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}
	else if(nProcs == 1)
	{
		if(rankL == 0)
		{	
			start_time = MPI_Wtime();
			if(malloc2dint(&resultMat, nrL1, nrC2) == 0)
			{
				for(i = 0; i < nrL1; i++)
					for(j = 0; j < nrC2; j++) 
					{
						resultMat[i][j] = 0;
						for(k = 0; k < nrC1; k++)
							resultMat[i][j] += mat1[i][k]*mat2[k][j];
					}
				stop_time = MPI_Wtime();
				printf("\nRuntime is: %f\n", stop_time - start_time);
				return resultMat;		
			}
			else
			{
				//Memory could not be allocated
				return NULL;
			}
		}
	}
	else
	{
		numTasksL = nProcs;
		//Array will keep the index of processes from the current group which will be added in a new group
		ranks = (int *) malloc(sizeof(int) * numTasksL);
		for(i = 0; i < numTasksL; i++)
		{
			ranks[i] = i;
		}
		if((testRes = MPI_Comm_group(MPI_COMM_WORLD, &group_world)) == MPI_SUCCESS)
		{
			//Defining a group with nProcs processes
			if((testRes = MPI_Group_incl(group_world, numTasksL, ranks, &resizedGroup)) == MPI_SUCCESS)
			{
				if((testRes = MPI_Comm_create(MPI_COMM_WORLD, resizedGroup, &resizedCommunicator)) == MPI_SUCCESS && rankL < nProcs)
				{
					if(malloc2dint(&resultMat, nrL1, nrC2) == 0)
					{
						//The parallel code begins here
						start_time = MPI_Wtime();		
						MPI_Comm_rank(resizedCommunicator, &rankL);
						
						//Array will keep the number of elements to be computed by each process
						toGet = (int *) malloc(sizeof(int) * numTasksL);
						//Array will keep the offsets relative to elements container address
						offsets = (int *) malloc(sizeof(int) * numTasksL);
	
						nrElem = n/numTasksL;	
						nrSupElem = n%numTasksL;
						
						//Defining offsets and number of elements to be sent to each process (information is available in every process)	
						for(i = 0; i < numTasksL; i++)
						{
							//For a distribution as balanced as possible, for the first processes a supplementary elem is considered
							//Normaly, each process will process elements
							if(i < nrSupElem)
							{
								//Send nrElem + 1
								toGet[i] = nrElem + 1;
							}
							else
							{
								//Send nrElem
								toGet[i] = nrElem;
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
						//Allocating space for the container of the computed elements
						resultElem = (int *) malloc(sizeof(int) * toGet[rankL]);
						//Alocating space to store the number of elements from first matrix to be sent to each process
						recevedLinesNumbers = (int *) malloc(sizeof(int) * numTasksL);

						//Alocating space for first matrix line offsets
						receivedLineOffsets = (int *) malloc(sizeof(int) * numTasksL);
						maxNumberOfLines = 1;
						//Lines indexes (offsets) and number of elements of matrix A to be sent to each process
						for(i = 0; i < numTasksL; i++)
						{
							int minLineIndex, maxLineIndex;
							minLineIndex = offsets[i] / nrC2;
							if(i == 0)
							{	
								maxLineIndex = (offsets[i + 1] - 1)/ nrC2;
								recevedLinesNumbers[i] = maxLineIndex;
								receivedLineOffsets[i] = minLineIndex * nrC1;
							}
							if(i == numTasksL - 1)
							{
								maxLineIndex = (n - 1) / nrC2;
								recevedLinesNumbers[i] = maxLineIndex - minLineIndex + 1;
								receivedLineOffsets[i] = minLineIndex * nrC1;
							}
							else
							{		
								maxLineIndex = (offsets[i + 1] - 1)/ nrC2;
								recevedLinesNumbers[i] = maxLineIndex - minLineIndex + 1;
								receivedLineOffsets[i] = minLineIndex * nrC1;
							}
							if(maxNumberOfLines < recevedLinesNumbers[i])
							{
								maxNumberOfLines = recevedLinesNumbers[i];
							}
							recevedLinesNumbers[i] *= nrC1;
						}
						//Allocating space for the matrix 1 elements to be received
						receivedLineElements = (int *) malloc(sizeof(int) * maxNumberOfLines * nrC1);
						//printf("\nProcess %d: Broadcasting...\n", rankL);
						//Broadcasting second matrix
						MPI_Bcast(&(mat2[0][0]), nrL2*nrC2, MPI_INT, 0, resizedCommunicator);
						//printf("\nProcess %d: Broadcast done!\n", rankL);

						//printf("\nProcess %d: Scattering...\n", rankL);
						MPI_Scatterv(&(mat1[0][0]), recevedLinesNumbers, receivedLineOffsets, MPI_INT, receivedLineElements, maxNumberOfLines * nrC1, MPI_INT, 0, resizedCommunicator);

						//Computing result elements
						for(i = 0; i < toGet[rankL]; i++)
						{
							int elem = 0, line_index, column_index;
							line_index = (offsets[rankL] + i) / nrC2;// computed element line index = line index in first matrix
							column_index = (offsets[rankL] + i) % nrC2; // computed element column index = column index in second matrix
							for(k = 0; k < nrC1; k++)
							{
								elem += receivedLineElements[line_index * nrC1 + k - receivedLineOffsets[rankL]] * mat2[k][column_index];
							}
							resultElem[i] = elem;
							//printf("\n(Process %d):  elem%d%d computed %d", rankL, line_index, column_index, elem);
						}

						//printf("\nProcess %d - \nLocal operations finished...\n", rank);
						MPI_Gatherv(resultElem, toGet[rankL], MPI_INT, &(resultMat[0][0]), toGet, offsets, MPI_INT, 0, resizedCommunicator);
						//printMatrix(resultMat, nrL1, nrC2);
						if(rankL == 0)
						{	
							stop_time = MPI_Wtime();
							printf("\nRuntime is: %f\n", stop_time - start_time);
						}
						free(toGet);
						free(offsets);
						free(resultElem);
						free(recevedLinesNumbers);
						free(receivedLineElements);
						free(receivedLineOffsets);
						free(ranks);					
						if (rankL == 0)
						{
							MPI_Comm_free(&resizedCommunicator);
							MPI_Group_free(&resizedGroup);
							MPI_Group_free(&group_world);
						}
						return resultMat;
					}
					else
					{
						//Memory could not be allocated
						return NULL;
					}
				}
				else
				{
					free(ranks);
					if(rankL == 0)
					{
						MPI_Group_free(&resizedGroup);
						MPI_Group_free(&group_world);
					}
					return NULL;
				}
			}
			else
			{
				free(ranks);
				if(rankL == 0)
				{
					MPI_Group_free(&group_world);
				}
				return NULL;
			}
		}
		else
		{
			free(ranks);
			return NULL;
		}
	}
	return NULL;
}

