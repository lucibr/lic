#include <float.h>

#include "memAlloc.h"
#include "constants.h"
#include "MPIWrapper.h"

typedef struct matPos
{
	int l, c;
} MATPOS;

//does not take into account the process of rank 0
//returns the MPI process rank
int mapElemProcess_FS(int matDim, int blockDim, int l, int c, int nProcs)
{
	int ownerProc, jump;
	jump = matDim/blockDim;
	ownerProc = (l*jump + c%jump)%nProcs + 1;
	return ownerProc;
}

//returns abolute index in a matrix based on the element position and matrix dimension
int matRelToAbsMatPos(int l, int c, int matDim)
{
	return l*matDim + c;
}

//returns a (line, column) pair corresponding to the given matrix absolute position
MATPOS matAbstoRelPos(int abs_pos, int matDim)
{
	MATPOS p;
	p.l = abs_pos/matDim;
	p.c = abs_pos%matDim;
	return p;
}

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

//System matrix is U-type matrix (upper triangular - all entries under its main diagonal are 0): solvable using back substitution
int serialBackSubst(double **mat, int matDim, double *b, int dim, double **result)
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
	for(i = matDim-1; i >= 0; i--)
	{
		auxSum = 0;
		for(j = matDim-1; j > i; j--)
		{
			auxSum += (mat[i][j]*(*result)[j]);
		}
		(*result)[i] = (b[i]-auxSum)/mat[i][i];
	}
	return 0;
}

int parallelForwardSubst(double **mat, int matDim, double *b, int dim, double **result, int blockDim, double *runtime, int nProcs)
{
	int rankL, i, j, aux, procMatDim, startElem, currentI, currentJ, p, receive, nrDiagElem = 0, r, proc, line,
		**procElemMatrix;
	double  toSend, auxF,
			*bValues, //will store the free term values processed locally; for the processes which do not store diagonal elements (neither free terms), 
			*xValues;//will store the free solution values processed locally
	PROCESSED_ELEM **localElem; //Will store the matrix elements processed locally by each process
	MATPOS currentPos, procPos;
	procMatDim = matDim/blockDim;
	
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(matDim % blockDim != 0)
	{	
		return -6;
	}

	if(procMatDim * procMatDim + 1 > nProcs)
	{
		return -3;
	}
	if(rankL == 0)
	{
		if(malloc2dint(&procElemMatrix, matDim, matDim) != 0)
		{
			if(rankL == 0)
			{
				printErrorMessage(-5, rankL, "parallelForwardSubst\0");
			}
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
		*result = (double *)calloc(matDim, sizeof(double));
		if(!result)
		{
			if(rankL == 0)
			{
				printErrorMessage(-5, rankL, "parallelForwardSubst\0");
			}
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
	}
	else
	{
		if(malloc2dPE(&localElem, blockDim, blockDim) != 0)
		{
			if(rankL == 0)
			{
				printErrorMessage(-5, rankL, "parallelForwardSubst\0");
			}
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
		bValues = (double *)malloc(blockDim * sizeof(double));
		xValues = (double *)calloc(blockDim, sizeof(double));
		if(!bValues || !xValues)
		{
			if(rankL == 0)
			{
				printErrorMessage(-5, rankL, "parallelForwardSubst\0");
			}
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}

		for(i = 0; i < blockDim; i++)
		{
			xValues[i] = DBL_MAX;
			for(j = 0; j < blockDim; j++)
			{
				localElem[i][j].computed = -1;
				localElem[i][j].value = 0;
			}
		}
	}
	if(rankL == 0)
	{
		startElem = 0;
		//Cyclick block distribution mapping (main proces (MPI rank 0) only)		
		while(startElem < matDim*matDim)
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
						procElemMatrix[currentI][currentJ] = mapElemProcess_FS(matDim, blockDim, currentI, currentJ, nProcs-1);
					}
					currentJ += (matDim/blockDim);
				}
				currentI += (matDim/blockDim);
				currentJ = (startElem-1)%matDim;
			}
		}
		//printMatrixInt(procElemMatrix, matDim, matDim);
		//Sending elements to be processed to workers
		for(i = 0; i < matDim; i++)
		{
			for(j = 0; j < matDim; j++)
			{
				if(j <= i)
				{
					aux = matRelToAbsMatPos(i, j, matDim);
					//Sending absolute position of the element (continue receive signal if positive)
					MPI_Send(&aux, 1, MPI_INT, procElemMatrix[i][j], 100, MPI_COMM_WORLD);
					//Sending element
					MPI_Send(&(mat[i][j]), 1, MPI_DOUBLE, procElemMatrix[i][j], 101, MPI_COMM_WORLD);
				}
			}
		}
		aux = -1;
		//Sending stop receiving matrix data to workers
		for(i = 1; i < nProcs; i++)
		{
			//Sending absolute position of the element (continue receive signal if positive) to workers	
			MPI_Send(&aux, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
		}
		//Sending free terms to workers
		for(i = 0; i < matDim; i++)
		{
			//Sending line/colummn index
			MPI_Send(&i, 1, MPI_INT, procElemMatrix[i][i], 222, MPI_COMM_WORLD);
			//Sending element
			MPI_Send(&(b[i]), 1, MPI_DOUBLE, procElemMatrix[i][i], 102, MPI_COMM_WORLD);
		}
	}
	else
	{
		//MATPOS currentPos;
		//Workers - receiving data to process
		//Receive matrix data
		aux = 0; //counts the number of received elems
		do
		{
			//Receive signal to continue
			MPI_Recv(&receive, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &status);
			if(receive >= 0)
			{	
				currentPos = matAbstoRelPos(receive, matDim);
				if(currentPos.l == currentPos.c)
				{
					nrDiagElem++;
				}			
				//Receive element
				MPI_Recv(&toSend, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &status);
				if(aux == 0)
				{
					localElem[0][0].value = toSend;
					localElem[0][0].computed = 1;
					localElem[0][0].absMatPos = receive;
				}
				else
				{
					int okC = 0, okL = 0;
					//Search local column where values from matrix absolute line are storred
					for(j = 0; j < blockDim; j++)
					{
						if(localElem[0][j].computed == 1 && matAbstoRelPos(localElem[0][j].absMatPos, matDim).c == currentPos.c)
						{
							okC = 1;
							break;
						}
					}
					//Search local line where values from matrix absolute line are storred
					for(i = 0; i < blockDim; i++)
					{
						if(localElem[i][0].computed == 1 && matAbstoRelPos(localElem[i][0].absMatPos, matDim).l == currentPos.l)
						{
							okL = 1;
							break;
						}
					}
					if(okC && okL)
					{
						localElem[i][j].value = toSend;
						localElem[i][j].computed = 1;
						localElem[i][j].absMatPos = receive; 
					}
					else if(okC)
					{
						//Elements from this line were not stored before
						for(i = 0; i < blockDim; i++)
						{
							if(localElem[i][j].computed != 1)
							{
								localElem[i][j].value = toSend;
								localElem[i][j].computed = 1;
								localElem[i][j].absMatPos = receive;
								break;
							}
						}
					}
					else if(okL)
					{
						//Elements from this column were not stored before
						for(j = 0; j < blockDim; j++)
						{
							if(localElem[i][j].computed != 1)
							{
								localElem[i][j].value = toSend;
								localElem[i][j].computed = 1;
								localElem[i][j].absMatPos = receive;
								break;
							}
						}
					}
					else
					{
						//Elements from this line/column were not stored before
						for(i = 0; i < blockDim; i++)
						{
							for(j = 9; j < blockDim; j++)
							{
								if(localElem[i][j].computed != 1)
								{
									localElem[i][j].value = toSend;
									localElem[i][j].computed = 1;
									localElem[i][j].absMatPos = receive;
									break;
								}
							}
						}
					}
					
				}
				aux++;
			}
		}
		while(receive >= 0);
		for(i = 0; i < nrDiagElem; i++)
		{
			//Receive line/column index
			MPI_Recv(&receive, 1, MPI_INT, 0, 222, MPI_COMM_WORLD, &status);
			for(j = 0; j < blockDim; j++)
			{
				currentPos = matAbstoRelPos(localElem[j][j].absMatPos, matDim);
				if(currentPos.c == receive && currentPos.l == receive)
				{
					MPI_Recv(&(bValues[j]), 1, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);
					break;
				}
			}
		}
//		if(rankL == 1)
//		{
//			for(i = 0; i < blockDim; i++)
//			{
//				for(j = 0; j < blockDim; j++)
//				{
//					printf("%.3lf (%d,%d)", localElem[i][j].value, matAbstoRelPos(localElem[i][j].absMatPos, matDim).l, matAbstoRelPos(localElem[i][j].absMatPos, matDim).c);
//				}
//				printf("\n");
//			}
//			for(i = 0; i < blockDim; i++)
//			{
//				printf("@@%.3lf ", bValues[i]);
//			}
//		}
	}

	//MPI_Barrier(MPI_COMM_WORLD);	
	//Data processing					
	if(rankL != 0)
	{
		if(nrDiagElem != 0)
		{
			for(i = 0; i < blockDim; i++)
			{
				for(j = 0; j < blockDim; j++)
				{
					if(localElem[i][j].computed == 1)
					{
						currentPos = matAbstoRelPos(localElem[i][j].absMatPos, matDim);
						if(currentPos.c == currentPos.l)
						{
													
							//If the processed element is of absolute index (0,0)
							if(currentPos.l == 0)
							{
								xValues[i] = bValues[i]/localElem[i][j].value;
								localElem[i][j].computed = 1;
								//Send computed solution to all column processes
								proc = rankL;
								for(r = 0; r < procMatDim; r++)
								{
									//printf("\n!?!??!?!?!!?Process %d: to process %d (r, blockdim) = (%d %d)...\n", rankL, xValues[i], proc, r, blockDim);
									//proc = mapElemProcess_FS(matDim, blockDim, currentPos.l+r, currentPos.c, nProcs-1);
									if(proc != rankL)
									{
										//Send solution component		
										//printf("\n!?!??!?!?!!?Process %d: Sendind solution component 0 (value %.3lf) to process %d...\n", rankL, xValues[i], proc);
										MPI_Send(&(xValues[i]), 1, MPI_DOUBLE, proc, currentPos.l, MPI_COMM_WORLD);
									}
									proc += procMatDim;
								}
							}
							else
							{	
								procPos = matAbstoRelPos(rankL-1, procMatDim);
								line = matAbstoRelPos(localElem[i][j].absMatPos, matDim).l; //Computing line for which partial sums are needed
								receive = 1;
								//Request sum elements from the row processes in ascending order of columns
								auxF = 0;
								for(r = 0; r < procMatDim; r++)
								{
									proc = rankL - procPos.c + r;
									//if(rankL == 1)
									//printf("\nProcess %d: request to %d...\n\n", rankL,proc);
									if(proc != rankL)
									{
										//Signal process proc to continue to process data from the current process
										MPI_Send(&receive, 1, MPI_INT, proc, 1111, MPI_COMM_WORLD);
										//Send to each row process the tag (line number) of the desired partial sum
										MPI_Send(&(line), 1, MPI_INT, proc, 111, MPI_COMM_WORLD);
										//if(rankL==1)
										//printf("\n>>Process %d: Requesting line %d partial sum from process %d...", rankL, line, proc);
										MPI_Recv(&toSend, 1, MPI_DOUBLE, proc, line, MPI_COMM_WORLD, &status);
										//if(rankL==1)
										//printf("\n<<Process %d: Partial sum (value %.3lf) for line %d received from process %d.", rankL, toSend, line, proc);
										auxF += toSend;
									}
								}
								//printf("\nProcess %d: toSend (before) = %.3lf.\n", rankL, toSend);
								xValues[i] = auxF;
								//Computing local partial sum corresponding to the line of the analized local element
								for(r = 0; r < j; r++)
								{
									if(localElem[i][r].computed == 1)
									{
										xValues[i] += (xValues[r]*localElem[i][r].value);
									}
								}
								xValues[i] = (bValues[i] - xValues[i])/localElem[i][j].value;
								//if(rankL==1)printf("\nProcess %d (): sending solution for line %d...\n",rankL, line);
								if(line < matDim - 1)
								{
									//Send solution to all column processes in ascending order of columns
									//Send solution to proceses on lines above the current process line
									if(j < blockDim-1)
									{
										r = procPos.l;
										proc = rankL;
										while(r > 0)
										{
											proc -= procMatDim;
											//Send solution component
											//printf("\n@@@Process %d: Sending solution component %d (value %.3lf) to process %d...", rankL, line, xValues[i], proc);
											MPI_Send(&(xValues[i]), 1, MPI_DOUBLE, proc, line, MPI_COMM_WORLD);
											r--;
										}

									}
									//Send solution to processes on lines under the current process line
									proc = rankL;
									for(r = procPos.l+1; r < procMatDim; r++)
									{								
										proc += procMatDim;		
										//Send solution component
										//printf("\n###Process %d: Sending solution component %d (value %.3lf) to process %d...", rankL, line, xValues[i], proc);
										MPI_Send(&(xValues[i]), 1, MPI_DOUBLE, proc, line, MPI_COMM_WORLD);										
									}
//									r = procPos.l;
//									proc = rankL;
//									while(r > 0)
//									{
//										proc -= procMatDim;
//										r--; 
//									}
//									printf("\nProcess %d: Starting r (%d) and proc %d...\n", rankL,r, proc);
//									for(p = r; p < procMatDim; p++)
//									{
//										if(proc!=rankL)
//										{
//										//Send solution component
//										printf("\n###Process %d: Sending solution component %d (value %.3lf) to process %d...", rankL, line, xValues[i], proc);
//										MPI_Send(&(xValues[i]), 1, MPI_DOUBLE, proc, line, MPI_COMM_WORLD);
//										}
//										proc+=procMatDim;
//									}
								}
							}
						}
					}
				}
			}
					
			//If diagonale process sending stop signal to all row processes
			procPos = matAbstoRelPos(rankL-1, procMatDim);
			if(procPos.l == procPos.c)
			{
				receive = 0;	
				for(r = 0; r < procMatDim; r++)
				{
					proc = rankL - procPos.c + r;
					if(proc != rankL)
					{
						//printf("\nProcess %d: sending signlae stop to process %d...\n", rankL, proc);
						//Signal process procMatrix[i][r] to continue to process data from the current process
						MPI_Send(&receive, 1, MPI_INT, proc, 1111, MPI_COMM_WORLD);
					}
				}
			}
		
			for(r = 0; r < blockDim; r++)
			{
				if(fabs(xValues[r] - DBL_MAX) > EPS)
				{
					line = matAbstoRelPos(localElem[r][r].absMatPos, matDim).l;
					MPI_Send(&(xValues[r]), 1, MPI_DOUBLE, 0, line, MPI_COMM_WORLD);
				}
			}
		}
		else
		{
			procPos = matAbstoRelPos(rankL-1, procMatDim);
			for(r = 0; r < blockDim; r++)
			{
				if(localElem[r][0].computed == 1 && matAbstoRelPos(localElem[r][0].absMatPos, matDim).c == 0)
				{
					MPI_Recv(&(xValues[0]), 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
					//printf("\n???Process %d: Received solution 0 (value %.3lf) -> %d from process 1.", rankL, xValues[0], 0);
					break;
				}
			}
			//Compute process rank which stores diag elements
			proc = rankL + (procPos.l-procPos.c);
			do
			{
				//Continue to process requests from the row process which storres diagonal elements? 
				//printf("\nProcess %d: Receveing signal from process %d...", rankL, proc);
				MPI_Recv(&receive, 1, MPI_INT, proc, 1111, MPI_COMM_WORLD, &status);
				//printf("\nProcess %d: receiived %d\n\n", rankL, receive);
				if(receive)
				{
					//Receive line for which partial sum will be returned
					MPI_Recv(&line, 1, MPI_INT, proc, 111, MPI_COMM_WORLD, &status);
					//printf("\nProcess %d: IdProc(%d) requested line %d partial sum...\n", rankL, proc, line);
					for(r = 0; r < blockDim; r++)
					{
						if(matAbstoRelPos(localElem[r][0].absMatPos, matDim).l == line)
						{
							break;
						}
					}
					if(r >= blockDim)
					{	
						toSend = 0;
						MPI_Send(&toSend, 1, MPI_DOUBLE, proc, line, MPI_COMM_WORLD);
					}
					else
					{
						//Receive all solutions elements needed for the partial sum from the column process
						for(p = 0; p < blockDim; p++)
						{
							if(localElem[r][p].computed > 0)
							{		
								int sourceProc, column;									
								column = matAbstoRelPos(localElem[r][p].absMatPos, matDim).c;
								if(fabs(xValues[p] - DBL_MAX) < EPS)
								{
									sourceProc = mapElemProcess_FS(matDim, blockDim, column, column, nProcs-1);
									//printf("\n#+#+#+Process %d: Requesting solution %d from process %d (tag %d)...", rankL, column, sourceProc, column);
									MPI_Recv(&(xValues[p]), 1, MPI_DOUBLE, sourceProc, column, MPI_COMM_WORLD, &status);	
									//printf("\n#-#-#-Process %d: Received solution %d (value = %.3lf) from process %d (tag %d)...", rankL, column, xValues[p], sourceProc, column);	
								}	
							}	
						}
						toSend = 0;
						for(p = 0; p < blockDim; p++)
						{	
							toSend += (xValues[p]*localElem[r][p].value);
						}
						//if(rankL==2)
						//printf("\n<<Process %d: Sending partial sum (value = %.3lf, tag = %d) to porcess %d...\n", rankL, toSend, line, proc);
						MPI_Send(&toSend, 1, MPI_DOUBLE, proc, line, MPI_COMM_WORLD);
					}
				}
			}
			while(receive);
		}
		//printf("\nProcess %d ending...\n",rankL);
	}
	else
	{
		int sourceProc;
		//Collecting results from workers
		if(rankL == 0)
		{
			for(i = 0; i < matDim; i++)
			{
				sourceProc = mapElemProcess_FS(matDim, blockDim, i, i, nProcs-1);
				MPI_Recv(&((*result)[i]), 1, MPI_DOUBLE, sourceProc, i, MPI_COMM_WORLD, &status);
			}
		}
	}

	//Free memory
	if(rankL == 0)
	{
		free2dint(&procElemMatrix);
	}
	else
	{
		free(bValues);
		free(xValues);
	
		free2dPE(&localElem);
	}

	return 0;
}

int parallelBackSubst(double **mat, int matDim, double *b, int dim, double **result, int blockDim, double *runtime, int nProcs)
{
	int rankL, i, j, aux, procMatDim, startElem, currentI, currentJ, continueAssignement, currentNrL, currentNrC, k, p, ok, receive, nrDiagElem = 0, r,
		localLineValue,
		**procMatrix, //Will store the processes indexes
		**procElemMatrix, //will store the process assignement for each elem (index of process + 1 -> process 1 is rank 0 process)
		*localLine, //will store, locally for each process, the absolute line indexes of received elements
		*localColumn, //will store, locally for each process, the absolute column indexes of received elements
		*computedX, //will store 1 in the corresponding position if the solution component was received/computed
		**receivedElem; //will store 1 in the corresponding position if the element was received
	double  toSend,
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
	localColumn = (int *)calloc(blockDim, sizeof(int));
	bValues = (double *)calloc(blockDim, sizeof(double));
	xValues = (double *)calloc(blockDim, sizeof(double));
	*result = (double *)calloc(matDim, sizeof(double));
	if(!localLine || !localColumn || !bValues || !xValues || !result || !computedX)
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
						if(procElemMatrix[currentI][currentJ] == 0 && currentJ >= currentI)
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
			if(procElemMatrix[i][i]-1 > 0)
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
	}			
	//Data processing
	ok = 1;
	for(i = 0; i < procMatDim; i++)
	{
		for(j =  0 ; j < procMatDim; j++)
		{
			if(procMatrix[i][j] == rankL)
			{
				if(i == j)
				{
					for(p = blockDim - 1; p >= 0; p--)
					{
						for(k = blockDim - 1; k >= 0; k--)
						{
							//Main diagonal task/element
							if(k == p && localLine[p] == localColumn[k])
							{
								if(receivedElem[p][k] == 1)
								{
									if(localLine[p] - 1 == matDim - 1)
									{
										xValues[p] = bValues[p]/localElem[p][k];
										(*result)[localLine[p]-1] = xValues[p];
										computedX[p] = 1;
										for(r = procMatDim - 1; r >= 0; r--)
										{
											//Send solution to all column processes
											if(procMatrix[r][j] != rankL)
											{
												//Send solution component												
												MPI_Send(&(xValues[p]), 1, MPI_DOUBLE, procMatrix[r][j], localLine[p]-1, MPI_COMM_WORLD);										
											}
										}
									}
									else
									{	
										receive = 1;
										//Request sum elements from the row processes in descending order of columns
										for(r = procMatDim - 1; r >= 0; r--)
										{
											if(procMatrix[i][r] != rankL)
											{
												//Signal process procMatrix[i][r] to continue to process data from the current process
												MPI_Send(&receive, 1, MPI_INT, procMatrix[i][r], 1111, MPI_COMM_WORLD);
												//Send to each row process the tag (line number) of the desired partial sum
												MPI_Send(&(localLine[p]), 1, MPI_INT, procMatrix[i][r], 111, MPI_COMM_WORLD);
												MPI_Recv(&toSend, 1, MPI_DOUBLE, procMatrix[i][r], localLine[p], MPI_COMM_WORLD, &status);
												xValues[p] += toSend;
											}
										}
										//Request all solutions elements needed from the column processes in ascending order of rows
										for(r = p+1; r < blockDim; r++)
										{
											if(receivedElem[p][r] == 1)
											{
												if(computedX[r] == 1)
												{
													xValues[p] += (xValues[r]*localElem[p][r]);
												}

											}
										}

										xValues[p] = (bValues[p] - xValues[p])/localElem[p][k];
										(*result)[localLine[p]-1] = xValues[p];
										computedX[p] = 1;
										if(localLine[p] < matDim)
										{
											for(r = procMatDim - 1; r >= 0; r--)
											{
												//Send solution to all column processes
												if(procMatrix[r][i] != rankL)
												{
													//Send solution component
													MPI_Send(&(xValues[p]), 1, MPI_DOUBLE, procMatrix[r][i], localLine[p]-1, MPI_COMM_WORLD);										
												}
											}
										}
									}
								}
							}
						}
						
					}
					receive = 0;	
					//Sending stop signal to all row processes
					for(r = procMatDim - 1; r >= 0; r--)
					{
						if(procMatrix[i][r] != rankL)
						{
							//Signal process procMatrix[i][r] to continue to process data from the current process
							MPI_Send(&receive, 1, MPI_INT, procMatrix[i][r], 1111, MPI_COMM_WORLD);
						}
					}
				}
				
				else
				{
					for(p = 0; p < blockDim; p++)
					{
						if(localColumn[p] == matDim)
						{
							MPI_Recv(&(xValues[p]), 1, MPI_DOUBLE, procElemMatrix[localColumn[k]-1][localColumn[k]-1]-1, localColumn[p]-1, MPI_COMM_WORLD, &status);
							computedX[p] = 1;
							break;
						}
					}
					do
					{
						//Continue to process requests from the row process which storres diagonal elements? 
						MPI_Recv(&receive, 1, MPI_INT, procMatrix[i][i], 1111, MPI_COMM_WORLD, &status);
						if(receive)
						{
							//Receive line for which partial sum will be returned
							MPI_Recv(&localLineValue, 1, MPI_INT, procMatrix[i][i], 111, MPI_COMM_WORLD, &status);
							for(p = 0; p < blockDim; p++)
							{
								if(localLine[p] == localLineValue)
								{
									break;
								}
							}
							if(p >= blockDim)
							{	
								toSend = 0;
								MPI_Send(&toSend, 1, MPI_DOUBLE, procMatrix[i][i], localLineValue, MPI_COMM_WORLD);
							}
							else
							{
								//Receive all solutions elements needed for the partial sum from the column process
								for(k = blockDim - 1; k >= 0; k--)
								{
									if(receivedElem[p][k] != 0)
									{											
										if(computedX[p] != 1)
										{
											MPI_Recv(&(xValues[p]), 1, MPI_DOUBLE, procElemMatrix[localColumn[k]-1][localColumn[k]-1]-1, localColumn[p]-1, MPI_COMM_WORLD, &status);
											computedX[p] = 1;		
										}	
									}	
								}
								toSend = 0;
								for(k = 0; k < blockDim; k++)
								{	
									toSend += (xValues[k]*localElem[p][k]);
								}	
								MPI_Send(&toSend, 1, MPI_DOUBLE, procMatrix[i][i], localLineValue, MPI_COMM_WORLD);
							}
						}
					}
					while(receive);
				}

			}
		}
	}

	//Collecting results
	if(rankL == 0)
	{
		for(i = 0; i < matDim; i++)
		{
			if(procElemMatrix[i][i]-1 != 0)
			{
				MPI_Recv(&((*result)[i]), 1, MPI_DOUBLE, procElemMatrix[i][i]-1, i, MPI_COMM_WORLD, &status);
			}
		}
	}
	else
	{
		for(r = 0; r < blockDim; r++)
		{
			if(localLine[r] == localColumn[r] && localLine[r] != 0)
			{
				MPI_Send(&(xValues[r]), 1, MPI_DOUBLE, 0, localLine[r]-1, MPI_COMM_WORLD);
			}
		}
	}
	//Free memory
	free(localLine);
	free(computedX);
	free(localColumn);
	free(bValues);
	free(xValues);
	
	free2ddouble(&localElem);
	free2dint(&procMatrix);
	free2dint(&procElemMatrix);
	free2dint(&receivedElem);

	return 0;
}

int forwardSubst(double **mat, int matDim, double *b, int dim, double **result, int blockDim, double *runtime, int nProcs)
{
	double start_time, stop_time;
	int rankL, res = -1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(matDim != dim)
	{
		return -13;
	}
	if(nProcs == 1  || matDim/blockDim == 1)
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

int backSubst(double **mat, int matDim, double *b, int dim, double **result, int blockDim, double *runtime, int nProcs)
{
	double start_time, stop_time;
	int rankL, res = -1;

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
			res = serialBackSubst(mat, matDim, b, dim, result);
			stop_time = MPI_Wtime();
			*runtime = stop_time - start_time;
		}
		return res;
	}
	else
	{
		start_time = MPI_Wtime();
		res = parallelBackSubst(mat, matDim, b, dim, result, blockDim, runtime, nProcs);
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return res;
	}
}
