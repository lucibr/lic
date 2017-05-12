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
	int rankL, i, j, aux, procMatDim, necProcs, isDiagonal = 0, procLine, procColumn, res, p, q,
		**procMatrix;
	double 	auxF,
			*dValues,
			**localElems,
			**localL,
			**auxContainer;

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
		
		printMatrixInt(procMatrix, procMatDim, procMatDim);
		printf("\nProcese necesare: %d", necProcs);
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
				//Receive partial sum block from process procMatrix[procLine][0]
				MPI_Recv(&(auxContainer[0][0]), blockDim*blockDim, MPI_DOUBLE, procMatrix[procLine][0], 2, MPI_COMM_WORLD, &status);

				//Send computed L-block to all inferior column processes
				//Send computed L-block to main process
			}
		}
		else
		{
			//Worker does not process a diagonal block
			if(malloc2ddouble(&auxContainer, blockDim, blockDim) != 0 || malloc2ddouble(&localL, blockDim, blockDim) != 0)
			{	
				printErrorMessage(-5, rankL, "parallelCholeskyFact_Block\0");
				MPI_Abort(MPI_COMM_WORLD, -5);
			}
			for(i = procColumn; i < procLine; i++)
			{
				//Receive necessary computation info from process above column process 
				MPI_Recv(&(auxContainer[0][0]), blockDim * blockDim, MPI_DOUBLE, procMatrix[i][procColumn], 1, MPI_COMM_WORLD, &status);
				if(i == procColumn)
				{
					//Information received is used to compute localL block which will be sent to process procMatrix[procLine][procLine]
				}
				else
				{
					//Information is processed and the result is sent to process procMatrix[procLine][procColumn + i]
				}
				//Send computed L-block to all inferior column processes
				//Send computed L-block to main process
			}
		}
		//Free localL, localElems, auxContainer
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
		res = parallelCholeskyFact(mat, matDim, L, blockDim, runtime, nProcs);
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return res;
	}
}
