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
				if(aux < 0)
				{
					//Matrix is not positive symmetric definite - Cholesky decomposition cannot be applied
					return -14;
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

//mat - matrix to be factorized
//matDim - matrix dimension (nrRows = nrColumns)
//blockDim - dimension of sub-matrix processed by each process
//runtime - execution duration
//nProcs - number of available processes
int parallelCholeskyFact_Block(double **mat, int matDim, double ***L, int blockDim, double *runtime, int nProcs)
{
	int rankL, i, j, aux, procMatDim, necProcs, isDiagonal = 0, procLine, procColumn, res, p, q, k,
		**procMatrix;
	double 	**localElems, auxF, 
			**localL,
			**auxContainer,
			**auxRowContainer;

	MPI_Status status;

	MPI_Datatype blockType2D;

	if(matDim % blockDim != 0)
	{	
		return -6;
	}
	procMatDim = matDim/blockDim;
	necProcs = (procMatDim * procMatDim - procMatDim)/2 + procMatDim + 1;
	if(necProcs > nProcs)
	{
		return -3;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	if(rankL >= necProcs)
	{
		return 0;
	}
	//Create MPI custom type in order to distribute to all processes 2D blocks of size dimR x dimC 
	//int MPI_Type_vector(int rowCount, int columnCount, int nrElemsToJump, MPI_Datatype oldtype, MPI_Datatype *newtype)
	if(MPI_Type_vector(blockDim, blockDim, matDim, MPI_DOUBLE, &blockType2D) != 0)
	{
		//Cannot create custom MPI type
		printErrorMessage(-7, rankL, "MPI_Type_vector->parallelCholeskyFact_Block\0");
		MPI_Abort(MPI_COMM_WORLD, -7);
	}
	MPI_Type_commit(&blockType2D);
	if(malloc2dint(&procMatrix, procMatDim, procMatDim) != 0)
	{	
		printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
		MPI_Abort(MPI_COMM_WORLD, -5);
	}
	if(rankL != 0)
	{
		if(malloc2ddouble(&localElems, blockDim, blockDim) != 0)
		{	
			printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
			MPI_Abort(MPI_COMM_WORLD, -5);
		}
	}
	aux = 1;
	//Creating mapping block-process matrix
	for(i = 0; i < procMatDim; i++)
	{
		for(j = 0; j <= i; j++)
		{
			procMatrix[i][j] = aux++;
			if(rankL == procMatrix[i][j])
			{
				procLine = i;
				procColumn = j;
				if(i == j)
				{
					isDiagonal = 1;
				}
			}
		}
	}
	if(rankL == 0)
	{
		//Main process
		if(malloc2ddouble(L, matDim, matDim) != 0)
		{	
			printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
			MPI_Abort(MPI_COMM_WORLD, -5);
		}
		
		//printMatrixInt(procMatrix, procMatDim, procMatDim);
		//printf("\nProcese necesare: %d", necProcs);
		//Main process - distributing data to workers 1...necProcs-1
		for(i = 0; i < procMatDim; i++)
		{
			for(j = 0; j <= i; j++)
			{
				if(i != j)
				{
					//Before sending info, check if original matrix is symmetric
					for(p = 0; p < blockDim; p++)
					{
						for(q = 0; q <= p; q++)
						{
							//TO CHECK IF!!!
							if(fabs(mat[i*blockDim + p][j*blockDim + q] - mat[j*blockDim + q][i*blockDim + p]) > EPS_CH_EQ)
							{
								printErrorMessage(-14, rankL, "parallelCholeskyFact_Block\0");
								MPI_Abort(MPI_COMM_WORLD, -14);
							}
						}
					}
				}
				else
				{
					//Before sending info, check if diagonal block is symmetric and positive definite
					for(p = 0; p < blockDim; p++)
					{
						for(q = 0; q <= p; q++)
						{
							//TO CHECK IF!!!
							if(p == q)
							{
								if(mat[i*blockDim + p][j*blockDim + q] < 0)
								{
									printErrorMessage(-14, rankL, "parallelCholeskyFact_Block\0");
									MPI_Abort(MPI_COMM_WORLD, -14);
								}
							}
							else if(fabs(mat[i*blockDim + p][j*blockDim + q] - mat[j*blockDim + q][i*blockDim + p]) > EPS_CH_EQ)
							{
								printErrorMessage(-14, rankL, "parallelCholeskyFact_Block\0");
								MPI_Abort(MPI_COMM_WORLD, -14);
							}
						}
					}
				}
				//Sending block to be processed to process procMatrix[i][j]
				MPI_Send(&(mat[i * blockDim][j * blockDim]), 1, blockType2D, procMatrix[i][j], 0, MPI_COMM_WORLD);
			}
		}
		//Collecting results from all worker processes
		for(i = 0; i < procMatDim; i++)
		{
			for(j = 0; j <= i; j++)
			{
				MPI_Recv(&((*L)[i * blockDim][j * blockDim]), 1, blockType2D, procMatrix[i][j], 111, MPI_COMM_WORLD, &status);
			}
		}
	}
	else
	{
		//Receiving matrix block to processed
		MPI_Recv(&(localElems[0][0]), blockDim * blockDim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		//WORKER process
		if(isDiagonal)
		{
			//Processes a diagonal block
			if(procLine == 0)
			{
				res = serialCholeskyFact(localElems, blockDim, &localL);	
				if(res != 0)
				{
					printErrorMessage(res, rankL, "parallelCholeskyFact_Block -> serialCholeskyFact\0");
					MPI_Abort(MPI_COMM_WORLD, res);
				}
				//Sending computed block to column processes
				for(i = 1; i < procMatDim; i++)
				{
					MPI_Send(&(localL[0][0]), blockDim*blockDim, MPI_DOUBLE, procMatrix[i][0], 1, MPI_COMM_WORLD);
				}
				//Sending computed L-block to main process (0)
				MPI_Send(&(localL[0][0]), blockDim*blockDim, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
			}
			else
			{
				if(malloc2ddouble(&localL, blockDim, blockDim) != 0 || malloc2ddouble(&auxContainer, blockDim, blockDim) != 0)
				{	
					printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
					MPI_Abort(MPI_COMM_WORLD, -5);
				}
				//Receive partial sums from left row processes
				for(i = 0; i < procColumn; i++)
				{
					//Receive partial sum block from process procMatrix[procLine][j]
					MPI_Recv(&(auxContainer[0][0]), blockDim*blockDim, MPI_DOUBLE, procMatrix[procLine][i], 2, MPI_COMM_WORLD, &status);
					for(p = 0; p < blockDim; p++)
					{
						for(q = 0; q <= p; q++)
						{
							localL[p][q] += auxContainer[p][q];
						}
					}
				}

				//Compute local L-block
				for(i = 0; i < blockDim; i++)
				{
					for(j = 0; j <= i; j++)
					{
						for(k = 0; k < j; k++)
						{
							localL[i][j] += (localL[i][k]*localL[j][k]);
						}
						if(i == j)
						{
							localL[i][j] = localElems[i][j] - localL[i][j];
							if(localL[i][j] < 0)
							{
								//Matrix is not positive symmetric definite - Cholesky decomposition cannot be applied
								printErrorMessage(-14, rankL, "parallelCholeskyFact_Block\0");
								MPI_Abort(MPI_COMM_WORLD, -14);
							}
							localL[i][j] = sqrt(localL[i][j]);
						}
						else
						{
							localL[i][j] = (localElems[i][j] - localL[i][j])/localL[j][j];
						}
					}
				}
				//Send computed L-block to all inferior column processes
				for(i = procLine+1; i < procMatDim; i++)
				{
					MPI_Send(&(localL[0][0]), blockDim*blockDim, MPI_DOUBLE, procMatrix[i][procColumn], 1, MPI_COMM_WORLD);
				}
				//Send computed L-block to main process
				MPI_Send(&(localL[0][0]), blockDim*blockDim, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
			}
		}
		else
		{
			//Worker does not process a diagonal block
			if(malloc2ddouble(&auxContainer, blockDim, blockDim) != 0 || malloc2ddouble(&localL, blockDim, blockDim) != 0 || malloc2ddouble(&auxRowContainer, blockDim, blockDim) != 0)
			{	
				printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
				MPI_Abort(MPI_COMM_WORLD, -5);
			}
			for(i = procColumn; i < procLine; i++)
			{
				//Receive necessary computation info from process above column process 
				MPI_Recv(&(auxContainer[0][0]), blockDim * blockDim, MPI_DOUBLE, procMatrix[i][procColumn], 1, MPI_COMM_WORLD, &status);
				//printMatrixDouble(auxContainer, blockDim, blockDim);
				if(i == procColumn)
				{
					//Receive info from the left row process
					for(j = 0; j < procColumn; j++)
					{
						//Receive partial sum values block from left row process
						MPI_Recv(&(auxRowContainer[0][0]), blockDim * blockDim, MPI_DOUBLE, procMatrix[procLine][j], 2, MPI_COMM_WORLD, &status);
						for(p = 0; p < blockDim; p++)
						{
							for(q = 0; q < blockDim; q++)
							{
								localL[p][q] += auxRowContainer[p][q];
							}
						}
					}
					//Information received is used to compute localL block
					for(p = 0; p < blockDim; p++)
					{
						for(q = 0; q < blockDim; q++)
						{
							for(k = 0; k < q; k++)
							{
								localL[p][q] += localL[p][k] * auxContainer[q][k];
							}
							localL[p][q] = (localElems[p][q] - localL[p][q])/auxContainer[q][q];
						}
					}
					//Computing partial sums for process procMatrix[procLine][procLine]
					for(p = 0; p < blockDim; p++)
					{
						for(q = 0; q <= p; q++)
						{
							auxContainer[p][q] = 0;
							for(k = 0; k < blockDim; k++)
							{
								auxContainer[p][q] += localL[p][k] * localL[q][k];
							}
						}
					}
					//Send partial sums to diagonal process on the  same line ???????????
					MPI_Send(&(auxContainer[0][0]), blockDim * blockDim, MPI_DOUBLE, procMatrix[procLine][procLine], 2, MPI_COMM_WORLD);
					//Send computed L-block to all inferior column processes
					for(j = procLine+1; j < procMatDim; j++)
					{
						MPI_Send(&(localL[0][0]), blockDim*blockDim, MPI_DOUBLE, procMatrix[j][procColumn], 1, MPI_COMM_WORLD);
					}
				}
				else
				{
					//Information is processed and the result is sent to right row processes starting with 'i' column in the procMatrix
					if(malloc2ddouble(&auxRowContainer, blockDim, blockDim) != 0)
					{	
						printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
						MPI_Abort(MPI_COMM_WORLD, -5);
					}
					//Compute partial sums
					for(p = 0; p < blockDim; p++)
					{
						for(q = 0; q < blockDim; q++)
						{
							for(k = 0; k < blockDim; k++)
							{
								auxRowContainer[p][q] += localL[p][k] * auxContainer[q][k];
							}
						}
					}
					//Send computed partial sums to right row porcesses 
					//for(j = i; j < procLine; j++)
					//{
						MPI_Send(&(auxRowContainer[0][0]), blockDim * blockDim, MPI_DOUBLE, procMatrix[procLine][i], 2, MPI_COMM_WORLD);
					//}
					free2ddouble(&auxRowContainer);
				}
			}
			//Send computed L-block to main process
			MPI_Send(&(localL[0][0]), blockDim*blockDim, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
			free2ddouble(&auxContainer);
		}
		//Free localL, localElems
		free2ddouble(&localElems);
		free2ddouble(&localL);
	}
	free2dint(&procMatrix);
	MPI_Type_free(&blockType2D);
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
		res = parallelCholeskyFact_Block(mat, matDim, L, blockDim, runtime, nProcs);
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return res;
	}
}
