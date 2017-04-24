#include <time.h>

#include "memAlloc.h"
#include "constants.h"
#include "MPIWrapper.h"

//System matrix is L-type matrix (lower triangular - all entries above its main diagonal are 0): solvable using forward substitution
int serialForwadSubst(double **mat, int matDim, double *b, int dim, double **result)
{
	int i, j;
	double auxSum;
	if(matDim != dim)
	{
		//incorrectly defined system
		return -13;
	}
	*result = (double *)calloc(matDim, sizeof(double));
	if(!result)
		return -5;
	for(i = 0; i < matDim; i++)
	{
		auxSum = 0;
		for(j = 0; j < i; j++)
		{
			auxSum += (mat[i][j]*(*result)[j]);
		}
		(*result)[i] = (b[i]-auxSum)/mat[i][i];
	}
	return 0;
}

int parallelForwardSubst(double **mat, int matDim, double *b, int dim, double **result, int blockDim, double *runtime, int nProcs)
{
	int rankL, i, j, aux, procMatDim, startElem, currentI, currentJ, continueAssignement, currentNrL, currentNrC, k, p, ok, receive, nrDiagElem = 0, r, hasADiagElem, isEmptyLine,
		**procMatrix, //Will store the processes indexes
		**procElemMatrix, //will store the process assignement for each elem (index of process + 1 -> process 1 is rank 0 process)
		*localLine, //will store, locally for each process, the absolute line indexes of received elements
		*localColumn, //will store, locally for each process, the absolute column indexes of received elements
		*computedX, //will store 1 in the corresponding position if the solution component was received/computed
		*requestedX, //will store 1 in the corresponding position if the solution component was requested
		**receivedElem; //will store 1 in the corresponding position if the element was received
	double  start_time, stop_time, toSend,
			*bValues, //will store the free term values processed locally
			*xValues, //will store the free solution values processed locally
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
		|| malloc2dint(&receivedElem, blockDim, blockDim) != 0)
	{
		if(rankL == 0)
		{
			printErrorMessage(-5, rankL, "parallelForwardSubst\0");
		}
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}

	localLine = (int *)calloc(blockDim, sizeof(int));
	computedX = (int *)calloc(blockDim, sizeof(int));
	requestedX = (int *)calloc(blockDim, sizeof(int));
	localColumn = (int *)calloc(blockDim, sizeof(int));
	bValues = (double *)calloc(blockDim, sizeof(double));
	xValues = (double *)calloc(blockDim, sizeof(double));
	if(!localLine || !localColumn || !bValues || !xValues)
	{
		if(rankL == 0)
		{
			printErrorMessage(-5, rankL, "parallelForwardSubst\0");
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
							continueAssignement = 0;
							procElemMatrix[currentI][currentJ] = procMatrix[i][j] + 1;
							//Counting the number of diagnonal elements that will be processed locally
							if(rankL == procMatrix[i][j] && currentI == currentJ)
							{
								nrDiagElem++;
							}
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
	
	//Data distribution to processes
	if(rankL == 0)
	{
		printf("\nMatricea proceselor:");
		printMatrixInt(procMatrix, procMatDim, procMatDim);
		printf("\nMatricea alocarii elementelor:");
		printMatrixInt(procElemMatrix, matDim, matDim);
//		printf("\n\nProcess %d - lines:\n", rankL+1);
//		printVectorInt(localLine, currentNrL);
//		printf("\n\nProcess %d - columns:\n", rankL+1);
//		printVectorInt(localColumn, currentNrC);

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
		
		//Send free terms
		for(i = 0; i < matDim; i++)
		{
			if(procElemMatrix[i][i]-1 != 0)
			{
				//Send diagonal index
				MPI_Send(&i, 1, MPI_INT, procElemMatrix[i][i]-1, 101, MPI_COMM_WORLD);
				//Send diagobal elem
				MPI_Send(&(b[i]), 1, MPI_DOUBLE, procElemMatrix[i][i]-1, 102, MPI_COMM_WORLD);
			}
			else
			{
				for(j = 0; j < currentNrL; j++)
				{
					if(localLine[j]-1 == i)
					{
						bValues[j]=b[i];
						break;
					}
				}
			}
		}
	}
	else
	{
//		printf("\n\nProcess %d - lines:\n", rankL+1);
//		printVectorInt(localLine, currentNrL);
//		printf("\n\nProcess %d - columns:\n", rankL+1);
//		printVectorInt(localColumn, currentNrC);
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
//		printf("\nProcess %d - local elements:\n", rankL+1);
//		printMatrixDouble(localElem, blockDim, blockDim);
		for(i = 0; i < nrDiagElem; i++)
		{
			//Receive diag index
			MPI_Recv(&aux, 1, MPI_INT, 0, 101, MPI_COMM_WORLD, &status);
			for(j = 0; j < currentNrL; j++)
			{
				if(localLine[j]-1 == aux)
				{
					MPI_Recv(&(bValues[j]), 1, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);
					break;
				}
			}
		}
//		printf("\nProcess %d - local free terms:\n", rankL+1);
//		printVectorDouble(bValues, nrDiagElem);
	}

	//Data processing
	ok = 1;
	for(i = 0; i < procMatDim; i++)
	{
		for(j = 0; j < procMatDim; j++)
		{
			if(procMatrix[i][j] == rankL)
			{
				for(p = 0; p < blockDim; p++)
				{
					for(k = 0; k < blockDim; k++)
					{
						//Main diagonal task/element
						if(rankL == 3)
						{
							printf("\n###Process %d: localElem(%d,%d)->globalElem(%d,%d)\n", rankL, p, k, localLine[p], localColumn[k]);
						}
						if(k == p && localLine[p] == localColumn[k])
						{
							if(receivedElem[p][k] == 1)
							{
								if(localLine[p] - 1 == 0)
								{
									xValues[p] = bValues[p]/localElem[p][k];
									computedX[p] = 1;
									for(r = 0; r < procMatDim; r++)
									{
										//Send solution to all column processes
										if(procMatrix[r][j] != rankL)
										{
											//Send column index
											//MPI_Send(&(localLine[p]), 1, MPI_DOUBLE, procMatrix[r][p], 105, MPI_COMM_WORLD);
											//Send solution component
											printf("\nProcess %d: Sending solution component %d (value %.3lf) to process %d (%d, %d)\n", rankL, localLine[p], xValues[p], procMatrix[r][j], r, j);
											MPI_Send(&(xValues[p]), 1, MPI_DOUBLE, procMatrix[r][j], 106, MPI_COMM_WORLD);										
										}
									}
								}
								else
								{	
									//Request sum elements from the row processes in ascending order of columns
									for(r = 0; r < procMatDim; r++)
									{
										if(procMatrix[i][r] != rankL)
										{
											printf("\nProcess %d: Receiving partial sum from process %d (%d, %d)\n", rankL, procMatrix[i][r], i, r);
											MPI_Recv(&toSend, 1, MPI_DOUBLE, procMatrix[i][r], 107, MPI_COMM_WORLD, &status);
											printf("\nProcess %d: Received partial sum (value %.3lf) from process %d (%d, %d)\n", rankL, toSend, procMatrix[i][r], i, r);
											xValues[p] += toSend;
										}
									}
									//Request all solutions elements needed from the column processes in ascending order of rows
									for(r = 0; r < blockDim; r++)
									{
										if(r != p)
										{
											if(computedX[r] == 1)
											{
												xValues[p] += (xValues[r]*localElem[p][r]);
											}
											else 
											{
												MPI_Recv(&(xValues[r]), 1, MPI_DOUBLE, procElemMatrix[localLine[r]-1][localLine[r]-1]-1, 108, MPI_COMM_WORLD, &status);
												computedX[r] = 1;
												xValues[p] += (xValues[r]*localElem[p][r]);
											}
										}
									}
									xValues[p] = (bValues[p] - xValues[p])/localElem[p][k];
									computedX[p] = 1;
									if(localLine[p] < matDim)
									{
										for(r = 0; r < procMatDim; r++)
										{
											//Send solution to all column processes
											if(procMatrix[r][i] != rankL)
											{
												//Send column index
//												MPI_Send(&(localLine[p]), 1, MPI_DOUBLE, procMatrix[r][p], 105, MPI_COMM_WORLD);
												//Send solution component
												MPI_Send(&(xValues[p]), 1, MPI_DOUBLE, procMatrix[r][i], 106, MPI_COMM_WORLD);										
											}
										}
									}
								}
							}
						}
						else
						{
							//Is there a diagonal element on the same line in the local matrix of the process
							
							hasADiagElem = 0;
							isEmptyLine = 1;
							for(r = 0; r < blockDim; r++)
							{
								if(receivedElem[p][r] != 0)
								{
									isEmptyLine = 0;
									break;
								}
							}
							if(isEmptyLine)
							{
								toSend = 0;
								k = blockDim;
								printf("\nProcess %d: Sending partial sum (value %.3lf) to process %d\n", rankL, toSend, procElemMatrix[localLine[p]-1][localLine[p]-1]);
								MPI_Send(&toSend, 1, MPI_DOUBLE, procElemMatrix[localLine[p]-1][localLine[p]-1], 107, MPI_COMM_WORLD);							
							}
							else
							{
								if(!computedX[k] && procElemMatrix[localColumn[k]-1][localColumn[k]-1]-1 != rankL && receivedElem[p][k] == 1)
								{
									printf("\nProcess %d: @@@ Receiving solution component %d from process %d\n", rankL, localColumn[k], procElemMatrix[localColumn[k]-1][localColumn[k]-1]-1);
									MPI_Recv(&(xValues[k]), 1, MPI_DOUBLE, procElemMatrix[localColumn[k]-1][localColumn[k]-1]-1, 106, MPI_COMM_WORLD, &status);
									printf("\nProcess %d: !!! Received solution component %d (value %.3lf) from process %d\n", rankL, localColumn[k], xValues[k], procElemMatrix[localColumn[k]-1][localColumn[k]-1]-1);
									computedX[k] = 1;
								}
									
								if(k == blockDim-1)
								{
									hasADiagElem = 0;
									for(r = 0; r < blockDim; r++)
									{
										if(localLine[r] == localColumn[r])
										{
											hasADiagElem = 1;
											break;
										}
									}
						
									if(!hasADiagElem)
									{
										//no diagonal elem storred locally on the same line
										toSend = 0;
										for(r = 0; r < blockDim; r++)
										{
											toSend += (localElem[p][r] * xValues[r]);
										}
										printf("\nProcess %d: Sending partial sum (value %.3lf) to process %d\n", rankL, toSend, procElemMatrix[localLine[p]-1][localLine[p]-1]-1);
										MPI_Send(&toSend, 1, MPI_DOUBLE, procElemMatrix[localLine[p]-1][localLine[p]-1]-1, 107, MPI_COMM_WORLD);
									}
								}
							}

						}



					}
				}
				ok = 0;
				break;
			}
		}
		if(!ok)
		{
			break;
		}
	}
	//Free memory
	
	return 0;
}

int forwardSubst(double **mat, int matDim, double *b, int dim, double **result, int blockDim, double *runtime, int nProcs)
{
	double start_time, stop_time;
	int rankL, res;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(matDim != dim)
	{
		return -13;
	}
	if(nProcs == 1)
	{
		if(rankL == 0)
		{
			start_time = MPI_Wtime();
			res = serialForwadSubst(mat, matDim, b, dim, result);
			stop_time = MPI_Wtime();
			*runtime = stop_time - start_time;
		}
		return res;
	}
	else
	{
		start_time = MPI_Wtime();
		res = parallelForwardSubst(mat, matDim, b, dim, result, blockDim, runtime, nProcs);
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return res;
	}
}
