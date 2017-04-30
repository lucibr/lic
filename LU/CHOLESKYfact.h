#include <math.h>

#include "memAlloc.h"
#include "constants.h"
#include "MPIWrapper.h"

#define EPS_CH_EQ 0.001

int serialCholeskyFact(double **mat, int matDim, double ***L)
{
	int i, j, k;
	double aux;
	if(malloc2ddouble(L, matDim, matDim) != 0)
	{	
		return -5;
	}
	for(i = 0; i < matDim; i++)
	{
		for(j = 0; j <= i; j++)
		{
			if(i == j)
			{
				//Main diagonal elements
				if(mat[i][j] < 0)
				{
					//Matrix is not positive symmetric definite - Cholesky decomposition cannot be applied
					return -14;
				}
				aux = mat[i][i];
				for(k = 0; k < i; k++)
				{
					aux -= ((*L)[i][k] * (*L)[i][k]);
				}
				(*L)[i][j] = sqrt(aux);
			}
			else
			{
				//Element under the main diagonal
				if(fabs(mat[i][j] - mat[j][i]) > EPS_CH_EQ)
				{
					//Matrix is not positive symmetric definite - Cholesky decomposition cannot be applied
					return -14;
				}
				aux = 0;
				for(k = 0; k < j; k++)
				{
					aux += (((*L)[i][k]) * ((*L)[j][k]));
				}
				(*L)[i][j] = (mat[i][j] - aux)/(*L)[j][j];
			}
		}
	}
	return 0;
}

int parallelCholeskyFact(double **mat, int matDim, double ***L, int blockDim, double *runtime, int nProcs)
{
	int rankL, i, j, aux, procMatDim, startElem, currentI, currentJ, continueAssignement, currentNrL, currentNrC, k, p, ok, receive, r,
		**procMatrix, //Will store the processes indexes
		**procElemMatrix, //will store the process assignement for each elem (index of process + 1 -> process 1 is rank 0 process)
		*localLine, //will store, locally for each process, the absolute line indexes of received elements
		*localColumn, //will store, locally for each process, the absolute column indexes of received elements
		*computedX, //will store 1 in the corresponding position if the solution component was received/computed
		**receivedElem; //will store 1 in the corresponding position if the element was received
	double  toSend,
			**localLvalues, //will store de solution components computed locally
			**localElem; //Will store the matrix elements processed locally by each process

	procMatDim = matDim/blockDim;
	
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(matDim % blockDim != 0)
	{	
		return -6;
	}

	if((matDim/blockDim) * (matDim/blockDim) > nProcs)
	{
		return -3;
	}

	if(malloc2ddouble(&localElem, blockDim, blockDim) != 0 || malloc2dint(&procMatrix, procMatDim, procMatDim) != 0 || malloc2dint(&procElemMatrix, matDim, matDim) != 0
		|| malloc2dint(&receivedElem, blockDim, blockDim) != 0 || malloc2ddouble(&localLvalues, blockDim, blockDim) != 0)
	{
		if(rankL == 0)
		{
			printErrorMessage(-5, rankL, "parallelCholeskyFact\0");
		}
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}

	localLine = (int *)calloc(blockDim, sizeof(int));
	computedX = (int *)calloc(blockDim, sizeof(int));
	localColumn = (int *)calloc(blockDim, sizeof(int));
	if(!localLine || !localColumn)
	{
		if(rankL == 0)
		{
			printErrorMessage(-5, rankL, "parallelCholeskyFact\0");
		}
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}

	//Cyclick block distribution mapping
	aux = startElem = currentNrL = currentNrC = 0;
	for(i = 0; i < procMatDim; i++)
	{
		for(j = 0; j < procMatDim; j++)
		{
			procMatrix[i][j] = aux++;
			continueAssignement = 1;
			while(continueAssignement && startElem < matDim*matDim)
			{
				currentI = startElem/matDim;
				currentJ = startElem%matDim;
				startElem++;
				while(currentI < matDim)
				{
					while(currentJ < matDim)
					{
						if(procElemMatrix[currentI][currentJ] == 0 && currentJ <= currentI)
						{

							if(rankL == 0) 
							{
								if((currentI == currentJ && mat[currentI][currentJ] < 0) || (currentI != currentJ && fabs(mat[currentI][currentJ] - mat[currentJ][currentI]) > EPS_CH_EQ))
								{
									printErrorMessage(-14, rankL, "parallelCholeskyFact\0");
									MPI_Abort(MPI_COMM_WORLD, -14);
								return -14;
								}
							}
								

							continueAssignement = 0;
							procElemMatrix[currentI][currentJ] = procMatrix[i][j] + 1;
							//Store the number of line/column where the current process has elements to process
							if(rankL == procMatrix[i][j])
							{
								ok = 1;
								for(k = 0; k < currentNrL; k++)
								{
									if(localLine[k] == currentI+1)
									{
										ok = 0;
										break;
									}
									if(!ok)
									{
										break;
									}
								}
								if(ok)
								{
									localLine[currentNrL++] = currentI+1;
								}
								ok = 1;
								for(k = 0; k < currentNrC; k++)
								{
									if(localColumn[k] == currentJ+1)
									{
										ok = 0;
										break;
									}
									if(!ok)
									{
										break;
									}
								}
								if(ok)
								{
									localColumn[currentNrC++] = currentJ+1;
								}
							}
						}
						currentJ += (matDim/blockDim);
					}
					currentI += (matDim/blockDim);
					currentJ = (startElem-1)%matDim;
				}	
			}
		}
	}
	
	if(rankL == 0)
	{
		printf("\nMatrix processes:\n");
		printMatrixInt(procMatrix, procMatDim, procMatDim);
		printf("\nProcess-element mapping matrix:\n");
		printMatrixInt(procElemMatrix, matDim, matDim);
	}

	//Data distribution to processes
	if(rankL == 0)
	{
		for(i = 0; i < matDim; i++)
		{
			for(j = 0; j < matDim; j++)
			{
				if(procElemMatrix[i][j]-1 == 0)
				{
					//Main process will not use MPI_Send/MPI_Receive but will directly store elements
					for(k = 0; k < blockDim; k++)
					{
						if(localLine[k]-1 == i)
						{
							for(p = 0; p < blockDim; p++)
							{
								if(localColumn[p]-1 == j)
								{
									localElem[k][p] = mat[i][j];
									receivedElem[k][p] = 1;
								}
							}
						}
					}
				}
				else if(procElemMatrix[i][j] != 0)
				{
					receive = 1;
					//Sending continue receive process signal to process procElemMatrix[i][j]-1
					MPI_Send(&receive, 1, MPI_INT, procElemMatrix[i][j]-1, 100, MPI_COMM_WORLD);
					//Sending line index of the element to send
					aux = i + 1;
					MPI_Send(&aux, 1, MPI_INT, procElemMatrix[i][j]-1, 1, MPI_COMM_WORLD);
					//Sending line index of the element to send
					aux = j + 1;
					MPI_Send(&aux, 1, MPI_INT, procElemMatrix[i][j]-1, 2, MPI_COMM_WORLD);
					//Sending the element to process procElemMatrix[i][j]-1
					MPI_Send(&(mat[i][j]), 1, MPI_DOUBLE, procElemMatrix[i][j]-1, 3, MPI_COMM_WORLD);
				}
			}
		}
		receive = 0;
		for(i = 1; i < procMatDim * procMatDim; i++)
		{
			MPI_Send(&receive, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
		}
	}
	else
	{
		do	
		{
			MPI_Recv(&receive, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &status);
			if(receive)
			{
				//Receive line number
				MPI_Recv(&i, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
				//printf("\nProcess %d: received line %d", rankL, i);
				//Receive column number
				MPI_Recv(&j, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
				//printf("\nProcess %d: received column %d", rankL, j);
				//Receive element
				MPI_Recv(&toSend, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
				//printf("\nProcess %d: received elem %.3lf\n", rankL, toSend);
				ok = 1;
				for(k = 0; k < currentNrL; k++)
				{
					if(i == localLine[k])
					{
						for(p = 0; p < currentNrC; p++)
						{
							if(j == localColumn[p])
							{
								localElem[k][p] = toSend;
								receivedElem[k][p] = 1;
								ok = 0;
								break;
							}
						}
					}
					if(!ok)
					{
						break;
					}
				}
			}
		}
		while(receive);
	}			

	//printf("\nProcess %d - received element:\n", rankL);
	//printMatrixDouble(localElem, blockDim, blockDim);

	//Free memory
	free(localLine);
	free(computedX);
	free(localColumn);

	free2ddouble(&localElem);	
	free2ddouble(&localLvalues);
	free2dint(&procMatrix);
	free2dint(&procElemMatrix);
	free2dint(&receivedElem);
	
	//printf("\n\n\nProcess %d: DONE!", rankL);
	return 0;
}

int choleskyFact(double **mat, int matDim, double ***L, int blockDim, double *runtime, int nProcs)
{
	double start_time, stop_time;
	int rankL, res = -1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(nProcs == 1 || matDim/blockDim == 1)
	{
		if(rankL == 0)
		{
			start_time = MPI_Wtime();
			res = serialCholeskyFact(mat, matDim, L);
			stop_time = MPI_Wtime();
			*runtime = stop_time - start_time;
		}
		return res;
	}
	else
	{
		start_time = MPI_Wtime();
		res = parallelCholeskyFact(mat, matDim, L, blockDim, runtime, nProcs);
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return res;
	}
}
