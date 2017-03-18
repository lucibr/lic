#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>
#include <time.h> 
#include <math.h>
#include <string.h>
#include <limits.h>


#define EPS 0.01

void printVectorInt(int *v, int dim)
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
			printf( "%.3lf ", m[i][j]);
		printf("\n");
	}
	printf("\n");
}

int malloc2dint(int ***array, int nrl, int nrc)
{
	int i;
	/* allocate the n*m contiguous items */
	int *p = (int *)calloc(nrl*nrc, sizeof(int));
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
	double *p = (double *)calloc(nrl*nrc, sizeof(double));
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

int infMatInv(double **mat, double **inv, int dim)
{
	int i, j, k;
	double multiplier, **aux;
	if(malloc2ddouble(&aux, dim, dim) != 0)
	{
		return -5;
	}
	//Initialize I matrix
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			aux[i][j] = mat[i][j];
			if(i == j)
			{
				inv[i][j] = 1;
			}
			else
			{
				inv[i][j] = 0;
			}
		}
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j <= i; j++)
		{
			if(i == j)
			{
				multiplier = aux[i][i];
				for(k = 0; k <= i; k++)
				{
					aux[i][k] /= multiplier;
					inv[i][k] /= multiplier;
				}
			}
			else
			{
				multiplier = aux[i][j];
				for(k = 0; k < i; k++)
				{
					aux[i][k] = aux[i][k] - multiplier * aux[j][k];
					inv[i][k] = inv[i][k] - multiplier * inv[j][k];
				}
			}
		}
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if((i == j && fabs(aux[i][j]-1) > EPS) || (i != j && fabs(aux[i][j]) > EPS))
			{
				return -2;
			}
		}
	}
	return 0;
}


int supMatInv(double **mat, double **inv, int dim)
{
	int i, j, k;
	double multiplier, **aux;
	if(malloc2ddouble(&aux, dim, dim) != 0)
	{
		return -5;
	}
	//Initialize I matrix
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			aux[i][j] = mat[i][j];
			if(i == j)
			{
				inv[i][j] = 1;
			}
			else
			{
				inv[i][j] = 0;
			}
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = dim - 1; j >= i; j--)
		{
			if(i == j)
			{
				multiplier = aux[i][i];
				for(k = dim - 1; k >= i; k--)
				{
					aux[i][k] /= multiplier;
					inv[i][k] /= multiplier;
				}
			}
			else
			{
				multiplier = aux[i][j];
				for(k = dim - 1; k > i; k--)
				{
					aux[i][k] = aux[i][k] - multiplier * aux[j][k];
					inv[i][k] = inv[i][k] - multiplier * inv[j][k];
				}
			}
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = dim - 1; j >= 0; j--)
		{
			if((i == j && fabs(aux[i][j]-1) > EPS) || (i != j && fabs(aux[i][j]) > EPS))
			{
				return -2;
			}
		}
	}
	return 0;
}

double** prodM_DI(double **mat1, int nrL1, int nrC1, int **mat2, int nrL2, int nrC2, int nProcs)
{
	int i, j, k;
	double **resultMat;
	if(nrC1 != nrL2)
	{
		//if(rankL == 0)
			printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}
	else if(nProcs == 1)
	{
		//if(rankL == 0)
		//{	
			if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
			{
				for(i = 0; i < nrL1; i++)
					for(j = 0; j < nrC2; j++) 
					{
						resultMat[i][j] = 0;
						for(k = 0; k < nrC1; k++)
							resultMat[i][j] += mat1[i][k]*((double)mat2[k][j]);
					}
				return resultMat;		
			}
			else
			{
				//Memory could not be allocated
				return NULL;
			}
		//}
	}
	return NULL;
}

double** prodM_ID(int **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, int nProcs)
{
	double **resultMat;
	int i, j, k;
	if(nrC1 != nrL2)
	{
		//if(rankL == 0)
			printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}
	else if(nProcs == 1)
	{
		//if(rankL == 0)
		//{	
			if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
			{
				for(i = 0; i < nrL1; i++)
					for(j = 0; j < nrC2; j++) 
					{
						resultMat[i][j] = 0;
						for(k = 0; k < nrC1; k++)
						{
							resultMat[i][j] += ((double)mat1[i][k])*mat2[k][j];
						}
					}
				return resultMat;		
			}
			else
			{
				//Memory could not be allocated
				return NULL;
			}
		//}
	}
	return NULL;
}

double** prodM_DD(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, int nProcs)
{
	double **resultMat;
	int i, j, k;
	if(nrC1 != nrL2)
	{
		//if(rankL == 0)
			printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}
	
	else if(nProcs == 1)
	{
		
		//if(rankL == 0)
		//{	
			if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
			{
				//printMatrixDouble(mat1, nrL1, nrC1);
				for(i = 0; i < nrL1; i++)
				{
					for(j = 0; j < nrC2; j++) 
					{
						for(k = 0; k < nrC1; k++)
						{
							resultMat[i][j] += mat1[i][k]*mat2[k][j];
						}
					}
				}
				//printMatrixDouble(resultMat, nrL1, nrC1);
				//printf("\nProdus returnat!");
				return resultMat;		
			}
			else
			{
				//Memory could not be allocated
				return NULL;
			}
		//}
	}
	return NULL;
}

int decompLU(double **mat, double **L, double **U, int **P, int nrL, int nrC, int nrLU, int nrCU, int nrLL, int nrCL, double **diagDiv)
{

	//Varianta neparalelizata LU = PA dreptunghiular
	//
	int i, j, k, a;
	int lineMaxPivot, dim = nrL;
	double maxPivot, aux;
	
	if(nrL > nrC)
	{
		dim = nrC;
	}
	for(i = 0; i < nrL; i++)
	{
		P[i][i] = 1;
	}
	//printf("\nMatricea initiala P:\n");
	//printMatrixInt(P, nrL, nrL);
	//printf("\nMatricea initiala L:\n");
	//printMatrixDouble(L, nrLL, nrCL);
	//printf("\nMatricea initiala U:\n");
	//printMatrixDouble(U, nrLU, nrCU);
	//printMatrixDouble(mat, nrL, nrC);
	
		
	for(i = 0; i < dim - 1; i++)
	{
		//Get the max value on the 'i' column (pivot value)
		lineMaxPivot = i;
		maxPivot = mat[i][i];
		//printf("\nInitial pivot %f", maxPivot);
		for(j = i+1; j < nrL; j++)
		{
			if(fabs(mat[j][i]) > fabs(maxPivot))
			{
				maxPivot = mat[j][i];
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
			//printf("\nInterschimbare linii %d si %d", i+1, lineMaxPivot+1);
			//The pivot value is not on the current line --> interchange the lines
			//Interchange lines in the original matrix
			for(j = 0; j < nrC; j++)
			{
				aux = mat[i][j];
				mat[i][j] = mat[lineMaxPivot][j];
				mat[lineMaxPivot][j] = aux;
			}
			for(j = 0; j < nrCL; j++)
			{
				aux = L[i][j];
				L[i][j] = L[lineMaxPivot][j];
				L[lineMaxPivot][j] = aux;
			}
			for(j = 0; j < nrL; j++)
			{
				a = P[i][j];
				P[i][j] = P[lineMaxPivot][j];
				P[lineMaxPivot][j] = a;  
			}
		}
		//printf("\nValoare pivot %f", maxPivot);
		// L coefficients (matricea multiplicatorilor)
		for(j = i; j < nrL; j++)
		{
			L[j][i] = mat[j][i]/maxPivot;
		}
		diagDiv[i][i] = 1/maxPivot;
		// U coefficients (matricea finala)
		if(i==0)
		{
			for(j = 0; j < nrC; j++)				
			{
				U[i][j] = mat[i][j];
			}		
		}
		for(j = i+1; j < nrL; j++)
			for(k = 0; k < nrC; k++)
			{
				mat[j][k] = mat[j][k] - L[j][i] * mat[i][k];
				
			}

		//printf("\nMatricea L - pas %d:\n", i + 1);
		//printMatrixDouble(L, nrLL, nrCL);
		//printf("\nMatricea P - pas %d:\n", i + 1);
		//printMatrixInt(P, nrL, nrL);
		//printf("\nMatrice - pas %d:\n", i + 1);
		//printMatrixDouble(mat, nrL, nrC);
	}
	for(i = 0; i < nrLU; i++)
	{
		for(j = 0; j < nrCU; j++)
		{
			U[i][j] = mat[i][j];
		}
	}
	


	if(nrL > nrC)
	{
		lineMaxPivot = nrC - 1;
		maxPivot = mat[nrC - 1][nrC - 1];
		//printf("\nInitial pivot %f", maxPivot);
		for(j = nrC; j < nrL; j++)
		{
			if(fabs(mat[j][i]) > fabs(maxPivot))
			{
				maxPivot = mat[j][i];
				lineMaxPivot = j;
			}			
		}
					
		if(fabs(maxPivot) <= EPS)
		{
			//pivot value to small
			return -1;
		}
		if(nrC - 1 != lineMaxPivot)
		{
			//printf("\nInterschimbare linii %d si %d", i+1, lineMaxPivot+1);
			//The pivot value is not on the current line --> interchange the lines
			//Interchange lines in the original matrix
			for(j = 0; j < nrC; j++)
			{
				aux = mat[nrC - 1][j];
				mat[nrC - 1][j] = mat[lineMaxPivot][j];
				mat[lineMaxPivot][j] = aux;

				aux = L[nrC - 1][j];
				L[nrC - 1][j] = L[lineMaxPivot][j];
				L[lineMaxPivot][j] = aux; 

				a = P[nrC - 1][j];
				P[nrC - 1][j] = P[lineMaxPivot][j];
				P[lineMaxPivot][nrC - 1] = a;
			}
		}
		//printf("\nValoare pivot %f", maxPivot);
		// L coefficients (matricea multiplicatorilor)
		for(j = nrC - 1; j < nrL; j++)
		{
			L[j][nrC - 1] = mat[j][nrC - 1]/maxPivot;
			if(j == nrC - 1) 
			{
				diagDiv[j][j] = 1/maxPivot;
			}
		}
	}
	else
	{
		L[nrL-1][nrL-1] = 1;
		diagDiv[nrL - 1][nrL - 1] = 1/mat[nrL - 1][nrL - 1];
	}
	//printf("\nMatricea U:\n");
	//printMatrixDouble(U, nrLU, nrCU);
	//printf("\nMatricea L:\n");
	//printMatrixDouble(L, nrLL, nrCL);
	return 0;
}

int parallelDecompLU(double **mat, double **L, double **U, int **P, int nrR, int nrC, int nrRU, int nrCU, int nrRL, int nrCL, int nProcs, int PR, int PC, int dimR, int dimC)
{
	if(nProcs == 1)
	{
		int res;
		double start_time, stop_time, **diagDiv;
		if(malloc2ddouble(&diagDiv, nrR, nrC) != 0)
		{
			//Memory allocation error
			return -5;
		}
		start_time = MPI_Wtime();
		res = decompLU(mat, L, U, P, nrR, nrC, nrRU, nrCU, nrRL, nrCL, diagDiv);
		stop_time = MPI_Wtime();
		printf("\nRuntime is: %f\n", stop_time - start_time);
		free2ddouble(&diagDiv);
		return res;
	}
	else if(nrR != nrC)
	{
		//Not square matrix
		return -2;
	} 
	else if(PC * PR > nProcs)
	{
		//Not sufficient resources for the specified processor grid
		return -3;
	}
	if(PC != PR || dimR != dimC)
	{
		//For square matrix the processes grid should also be square and the block sizes should be the equals on both directions
		return -4;	
	}
	else if(nrR % dimR != 0 || nrC % dimC != 0)
	{
		//Process grid cannot be matched with the matrix size
		return -6;
	}
	//printf("\nStarted parallel LU!!!\n");
	int **processes, rankL, *blocksPerProcess;	
	int dimRP = nrR/dimR, dimCP = nrC/dimC; //dimensions of processes matrix	
	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(malloc2dint(&processes, dimRP, dimCP) == 0)
	{
		MPI_Status status;
		MPI_Request request, requestL, requestP, requestU, requestMC, requestMR;	
		double start_time, stop_time;
		int i, j, k, p, q;
		MPI_Datatype blockType2D, blockType2DINT;
		blocksPerProcess = (int *) calloc(nProcs, sizeof(int));
		if(blocksPerProcess == NULL)
		{
			//Memory allocation error
			return -5;
		}
		start_time = MPI_Wtime();
		//Create the processes matrix
		for(p = 0; p < dimRP; p+=PR)
		{
			for(q = 0; q < dimCP; q+=PC)
			{
				k = 0;	
				for(i = 0; i < PR; i++)
				{
					while(k % PC != 0)
					{
						k++;
					} 
					for(j = 0; j < PC; j++)
					{
						if(i+p < dimRP && j+q < dimCP)
						{
							blocksPerProcess[nProcs - k - 1]++;
							processes[i+p][j+q] = nProcs - 1 - k++;
						}
					}
				}
			}
		}
		//Create MPI custom type in order to distribute to all processes 2D blocks of size dimR x dimC 
		//int MPI_Type_vector(int rowCount, int columnCount, int nrElemsToJump, MPI_Datatype oldtype, MPI_Datatype *newtype)
		if(MPI_Type_vector(dimR, dimC, nrC, MPI_DOUBLE, &blockType2D) != 0)
		{
			//Cannot create custom MPI type
			return -7;
		}
		MPI_Type_commit(&blockType2D);
		if(rankL == 0)
		{
			//printf("\nCustom type created!!!\n");
			printf("\nMatrix of processes:\n");
			printMatrixInt(processes, dimRP, dimCP);
			printVectorInt(blocksPerProcess, nProcs);
			//printf("\nProcess %d: Sending blocks to each process...\n", rankL);
			k = 0;
			for(i = 0; i < dimRP; i++)
			{
				for(j = 0; j < dimCP; j++)
				{
					//Main process will not send data to itself - communication not needed since it stores the whole matrix
					if((p = processes[i][j]) != 0)
					{
						int currentBlockIndex = i * dimCP + j;
						//currentBlockStartOffset = i * dimR * nrC + j * dimC;
						//Send block index
						//printf("\nProcess %d: Sending block index %d (start offset %d) to process %d...\n", rankL, currentBlockIndex, currentBlockStartOffset, p);
						MPI_Send(&currentBlockIndex, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
						//Send block of dimR x dimC size
						MPI_Isend(&(mat[i * dimR][j * dimC]), 1, blockType2D, p, 0, MPI_COMM_WORLD, &request);
						//printf("\nProcess %d: Block %d sent to process %d...\n", rankL, currentBlockIndex, p);
					}
				}
			}
		}
		int *localBlockIdexes,
				**localPBlocks[blocksPerProcess[rankL]]; //Stores the P values
		double	**localBlocks[blocksPerProcess[rankL]], //Stores the actual matrix values; after processing will store U values
				**localLBlocks[blocksPerProcess[rankL]]; //Will store the L values
		if(rankL != 0)
		{
			int nrBlocks = blocksPerProcess[rankL];
			localBlockIdexes = (int *) calloc(nrBlocks, sizeof(int));
			for(i = 0; i < nrBlocks; i++)
			{
				MPI_Recv(&(localBlockIdexes[i]), 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
				//printf("\nProcess %d: block index %d received!", rankL, localBlockIdexes[i]);
				if(malloc2ddouble(&(localBlocks[i]), dimR, dimC) != 0 || malloc2ddouble(&(localLBlocks[i]), dimR, dimC) != 0 || malloc2dint(&(localPBlocks[i]), dimR, dimC) != 0)
				{
					//Memory allocation error
					return -5;
				}
				MPI_Recv(&((localBlocks[i])[0][0]), dimR*dimC, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				//printf("\nProcess %d: block %d received.", rankL, localBlockIdexes[i]);
				//printf("\n!!!Process %d: %d", rankL, &((localBlocks[i])[0][0]) );
				//printf("\n!!!Process %d: %.3lf", rankL, (localBlocks[i])[0][0] );
				//printMatrixDouble(localBlocks[i], dimR, dimC);
				
			}
		}
		double	**localAux, **multiplierR, **multiplierC; //temporary dimR * dimC 2D array
		int **localP;
		for(i = 0; i < dimRP; i++)
		{
			for(j = 0; j < dimCP; j++)
			{
				if(processes[i][j] == rankL)
				{
					//computing index of local block to be processed
					int globalIndexOfBlockToProcess = i * dimCP + j, res = -1, nrBlocks, localIndexOfBlockToProcess = -1;
					nrBlocks = blocksPerProcess[rankL];
					//printVectorInt(localBlockIdexes, nrBlocks);


					if(malloc2ddouble(&localAux, dimR, dimC) != 0 || malloc2ddouble(&multiplierR, dimR, dimC) != 0 || malloc2ddouble(&multiplierC, dimR, dimC) != 0)
					{
						//Memory allocation error
						if(rankL != 0)
						{
							free(localBlockIdexes);
							for(p = 0; p < nrBlocks; p++)
							{
								free2ddouble(&(localBlocks[p]));
								free2ddouble(&(localLBlocks[p]));
								free2dint(&(localPBlocks[p]));	
							}
						}
						free(blocksPerProcess);
						MPI_Type_free(&blockType2D);
						free2dint(&processes);
						return -5;
					}


					if(i == j)
					{
						if(rankL != 0)
						{
							for(p = 0; p < nrBlocks; p++)
							{
								//printf("\n@@@Process %d: block index %d @@@ (searched %d)", rankL, localBlockIdexes[p], globalIndexOfBlockToProcess);
								if(localBlockIdexes[p] == globalIndexOfBlockToProcess)
								{
									localIndexOfBlockToProcess = p;
									break;
								}
							}
							//printf("\nProcess %d will call LU for local block %d (global index %d)...", rankL, localIndexOfBlockToProcess, globalIndexOfBlockToProcess);
							//printf("\nProcess %d - Matricea:\n", rankL);
							//printMatrixDouble(localBlocks[localIndexOfBlockToProcess], dimR, dimC);
							res = decompLU(localBlocks[localIndexOfBlockToProcess], localLBlocks[localIndexOfBlockToProcess], localAux, localPBlocks[localIndexOfBlockToProcess], dimR, dimC, dimR, dimC, dimR, dimC, multiplierC);
							//printf("\nProcess %d - Matricea L:\n", rankL);
							//printMatrixDouble(localLBlocks[localIndexOfBlockToProcess], dimR, dimC);
							if (rankL == 8)
							{
								printMatrixDouble(multiplierC, dimR, dimC);
							}
							//printMatrixDouble(multiplier, dimR, dimC);
							//printf("\nProcess %d - Matricea U:\n", rankL);
							//printMatrixDouble(localAux, dimR, dimC);
							//printf("\nProcess %d - Matricea:\n", rankL);
							//printMatrixDouble(localBlocks[localIndexOfBlockToProcess], dimR, dimC);
							if(res != 0)
							{
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);
								free(localBlockIdexes);
								for(p = 0; p < nrBlocks; p++)
								{
									free2ddouble(&(localBlocks[p]));
									free2ddouble(&(localLBlocks[p]));
									free2dint(&(localPBlocks[p]));	
								}
								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return res;
							}
							//Send to the multiplier to all column processes
							//printMatrixDouble(multiplier, dimR, dimC);
							for(p = i+1; p < dimCP; p++)
							{
								//printf("\nProcess %d: Sending multiplier to column process %d...", rankL, processes[p][j]);
								//MPI_Isend(&(multiplierC[0][0]), dimR * dimC, MPI_DOUBLE, processes[p][j], 2, MPI_COMM_WORLD, &requestMC);
								MPI_Send(&(multiplierC[0][0]), dimR * dimC, MPI_DOUBLE, processes[p][j], 2, MPI_COMM_WORLD);
							}
							//Computing multiplier for row processes Linv * P
								//Computing Linv
							if(infMatInv(localLBlocks[localIndexOfBlockToProcess], localAux, dimR) != 0)
							{
								//Cannot invert inferior triangular matrix
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);

								free(localBlockIdexes);
								for(p = 0; p < nrBlocks; p++)
								{
									free2ddouble(&(localBlocks[p]));
									free2ddouble(&(localLBlocks[p]));	
									free2dint(&(localPBlocks[p]));
								}

								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -8;
							}
							//printMatrixDouble(localAux, dimR, dimC);
							//Computing Linv * P
							multiplierR = prodM_DI(localAux, dimR, dimC, localPBlocks[localIndexOfBlockToProcess], dimR, dimC, 1);
							//printMatrixDouble(multiplierR, dimR, dimC); //DE TRIMIS SEPARAT P SI L
							if(multiplierR == NULL)
							{
								//Error in matrix product	
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);

								free(localBlockIdexes);
								for(p = 0; p < nrBlocks; p++)
								{
									free2ddouble(&(localBlocks[p]));
									free2ddouble(&(localLBlocks[p]));	
									free2dint(&(localPBlocks[p]));
								}

								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -9;
							}
							//Send multiplier to all row processes
							for(p = j+1; p < dimCP; p++)
							{
								//printf("\nProcess %d: Sending multiplier to row process %d...", rankL, processes[i][p]);
								//MPI_Isend(&(multiplierR[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][p], 3, MPI_COMM_WORLD, &requestMR);
								MPI_Send(&(multiplierR[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][p], 3, MPI_COMM_WORLD);
							}
							int tag = globalIndexOfBlockToProcess;
							
							//Sending computed results to main process (0) tag 100, + computed index
							//Sending computed index
							//printf("\nProcess %d: Sending computed index block %d to process 0...", rankL, globalIndexOfBlockToProcess);
							//MPI_Isend(&globalIndexOfBlockToProcess, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &request);
							MPI_Send(&globalIndexOfBlockToProcess, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
							//Sending U
							printf("\nProcess %d: Sending U to process 0...", rankL);
							//MPI_Isend(&((localBlocks[localIndexOfBlockToProcess])[0][0]), dimR * dimC, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &requestU);
							MPI_Send(&((localBlocks[localIndexOfBlockToProcess])[0][0]), dimR * dimC, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
							//Sending L
							printf("\nProcess %d: Sending L to process 0...", rankL);
							//MPI_Isend(&((localLBlocks[localIndexOfBlockToProcess])[0][0]), dimR * dimC, MPI_DOUBLE, 0, tag/10, MPI_COMM_WORLD, &requestL);
							MPI_Send(&((localLBlocks[localIndexOfBlockToProcess])[0][0]), dimR * dimC, MPI_DOUBLE, 0, tag/10, MPI_COMM_WORLD);
							//printMatrixDouble(localLBlocks[localIndexOfBlockToProcess], dimR, dimC);
							//Sending P
							printf("\nProcess %d: Sending P to process 0...", rankL);
							//MPI_Isend(&((localPBlocks[localIndexOfBlockToProcess])[0][0]), dimR * dimC, MPI_INT, 0, tag/100, MPI_COMM_WORLD, &requestP);
							MPI_Send(&((localPBlocks[localIndexOfBlockToProcess])[0][0]), dimR * dimC, MPI_INT, 0, tag/100, MPI_COMM_WORLD);
							printf("\nProcess %d: INFO sent to process 0.", rankL);

							//MPI_Wait(&requestMC, &status);
							//MPI_Wait(&requestMR, &status);
						}


						else
						{
							//Process 0 sholud process data taken directly from matrix mat 					
							double	**localA, **localL;
							//int currentBlockStartOffset = i * dimR * nrC + j * dimC;
							if(malloc2ddouble(&localA, dimR, dimC) != 0 || malloc2ddouble(&localL, dimR, dimC) != 0 || malloc2dint(&localP, dimR, dimC) != 0)
							{
								free(blocksPerProcess);
								free2ddouble(&localAux);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -5;
							}
							for(p = 0; p < dimR; p++)
							{
								for(q = 0; q < dimC; q++)
								{
									localA[p][q] = mat[i * dimR + p][j * dimC + q];
								}
							}
							//TO CHECK MORE IF THINGS DO NOT GO RIGHT
							//printf("\nProcess %d - Matricea:\n", rankL);
							//printMatrixDouble(localA, dimR, dimC);
							res = decompLU(localA, localL, localAux, localP, dimR, dimC, dimR, dimC, dimR, dimC, multiplierC);
							
							//printf("\nProcess %d - Matricea L:\n", rankL);;
							//printMatrixDouble(localL, dimR, dimC);
							//printf("\nProcess %d - Matricea U:\n", rankL);
							//printMatrixDouble(localAux, dimR, dimC);
							//printf("\nProcess %d - Matricea L:\n", rankL);
							//	printMatrixInt(localP, dimR, dimC);
							if(res != 0)
							{
								free2ddouble(&localA);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2dint(&localP);
								free2ddouble(&localL);
								free2ddouble(&localAux);
								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return res;
							}

							//Copy results into position in the final result containers L, U, P
							for(p = 0; p < dimR; p++)
							{
								for(q = 0; q < dimC; q++)
								{
									L[i * dimR + p][j * dimC + q] = localL[p][q];
									U[i * dimR + p][j * dimC + q] = localAux[p][q];
									P[i * dimR + p][j * dimC + q] = localP[p][q];
								}
							}
							
							//Send to the multiplier to all column processes
							//printMatrixDouble(multiplier, dimR, dimC);
							for(p = i+1; p < dimCP; p++)
							{
								//printf("\nProcess %d: Sending multiplier to column process %d...", rankL, processes[p][j]);
								MPI_Isend(&(multiplierC[0][0]), dimR * dimC, MPI_DOUBLE, processes[p][j], 2, MPI_COMM_WORLD, &requestMC);
							}
							
							//Computing multiplier for row processes Linv * P
								//Computing Linv
							if(infMatInv(localL, localAux, dimR) != 0)
							{
								//Cannot invert inferior triangular matrix
								free2ddouble(&localA);
								free2ddouble(&localL);
								free2dint(&localP);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);
								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -8;
							}
							//printMatrixDouble(localAux, dimR, dimC);
							//Computing Linv * P
							multiplierR = prodM_DI(localAux, dimR, dimC, localP, dimR, dimC, 1);
							//printMatrixDouble(multiplierR, dimR, dimC);
							if(multiplierR == NULL)
							{
								//Error in matrix product
								free2ddouble(&localA);
								free2ddouble(&localL);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);
								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -9;
							}

							//Send multiplier to all row processes
							for(p = j+1; p < dimCP; p++)
							{
								//printf("\nProcess %d: Sending multiplier to row process %d...", rankL, processes[i][p]);
								MPI_Isend(&(multiplierR[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][p], 3, MPI_COMM_WORLD, &requestMR);

							}
						}	
						
					}




					else if(i > j)
					{
						//This is a column process
						
						//The process should receive the multiplier from the idicated by processes[j][j]
						//printf("\nProcess %d: Receiving column multiplier from process %d...", rankL, processes[j][j]);
						MPI_Recv(&(multiplierC[0][0]), dimR * dimC, MPI_DOUBLE, processes[j][j], 2, MPI_COMM_WORLD, &status);
						//printMatrixDouble(multiplierC, dimR, dimC);
						//printf("\nProcess %d: Column multiplier received from process %d...", rankL, processes[j][j]);
						if (rankL == 0)
						{
							//Process 0 sholud process data taken directly from matrix mat 					
							double	**localA;
							if(malloc2ddouble(&localA, dimR, dimC) != 0)
							{
								free(blocksPerProcess);
								free2ddouble(&localAux);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -5;
							}
							//Copy the block that will be processed
							for(p = 0; p < dimR; p++)
							{
								for(q = 0; q < dimC; q++)
								{
									localA[p][q] = mat[i * dimR + p][j * dimC + q];
								}
							}
							//MPI_Wait(&request, &status);
							//printMatrixDouble(multiplierC, dimR, dimC);
							localAux = prodM_DD(localA, dimR, dimC, multiplierC, dimR, dimC, 1);

							if (localAux == NULL)
							{
								//Error in matrix product
								free2ddouble(&localA);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);
								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -9;
							}
				
							//Copy results into position in the final result container L
							for(p = 0; p < dimR; p++)
							{
								for(q = 0; q < dimC; q++)
								{
									L[i * dimR + p][j * dimC + q] = localAux[p][q];
								}
							}
						}

						else
						{
							//Get the local index of the block to be processed
							for(p = 0; p < nrBlocks; p++)
							{
								//printf("\n@@@Process %d: block index %d @@@ (searched %d)", rankL, localBlockIdexes[p], globalIndexOfBlockToProcess);
								if(localBlockIdexes[p] == globalIndexOfBlockToProcess)
								{
									localIndexOfBlockToProcess = p;
									break;
								}
							}

							//printMatrixDouble(localBlocks[localIndexOfBlockToProcess], dimR, dimC);
							//printMatrixDouble(multiplierC, dimR, dimC);
							localAux = prodM_DD(localBlocks[localIndexOfBlockToProcess], dimR, dimC, multiplierC, dimR, dimC, 1);
							//printMatrixDouble(localAux, dimR, dimC);
							if (localAux == NULL)
							{
								//Error in matrix product
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);

								free(localBlockIdexes);
								for(p = 0; p < nrBlocks; p++)
								{
									free2ddouble(&(localBlocks[p]));
									free2ddouble(&(localLBlocks[p]));	
									free2dint(&(localPBlocks[p]));
								}

								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								printf("\nProcess %d: Matrix product error!!!", rankL);
								return -9;
							}

							//Sending computed results to main process (0) tag 101, + computed index
							//Sending computed index
							//printf("\nProcess %d: Sending info to process 0...", rankL);
							//MPI_Isend(&globalIndexOfBlockToProcess, 1, MPI_INT, 0, 101, MPI_COMM_WORLD, &request);
							MPI_Send(&globalIndexOfBlockToProcess, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);
							//Sending L
							
							//printMatrixDouble(localAux, dimR, dimC);
							MPI_Send(&(localAux[0][0]), dimR * dimC, MPI_DOUBLE, 0, globalIndexOfBlockToProcess, MPI_COMM_WORLD);	
							//MPI_Isend(&(localAux[0][0]), dimR * dimC, MPI_DOUBLE, 0, globalIndexOfBlockToProcess, MPI_COMM_WORLD, &requestL);
							
							//MPI_Wait(&request, &status);
							//MPI_Wait(&requestL, &status);
							printf("\nProcess %d: L block sent to process 0.", rankL);
						}
						
					}




					else
					{
						//i < j -> This is a row process
						//The process should receive the multiplier from the idicated by processes[i][i]
						//printf("\nProcess %d: Receiving row multiplier from process %d...", rankL, processes[i][i]);
						MPI_Recv(&(multiplierR[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][i], 3, MPI_COMM_WORLD, &status);
						//printf("\nProcess %d: Row multiplier received from process %d...", rankL, processes[i][i]);
						//printMatrixDouble(multiplierR, dimR, dimC);
						if (rankL == 0)
						{
							//Process 0 sholud process data taken directly from matrix mat 					
							double	**localA;
							if(malloc2ddouble(&localA, dimR, dimC) != 0)
							{
								free(blocksPerProcess);
								free2ddouble(&localAux);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -5;
							}
							//Copy the block that will be processed
							for(p = 0; p < dimR; p++)
							{
								for(q = 0; q < dimC; q++)
								{
									localA[p][q] = mat[i * dimR + p][j * dimC + q];
								}
							}

							localAux = prodM_DD(multiplierR, dimR, dimC, localA, dimR, dimC, 1);

							if (localAux == NULL)
							{
								//Error in matrix product
								free2ddouble(&localA);
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);
								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -9;
							}
				
							//Copy results into position in the final result container L
							for(p = 0; p < dimR; p++)
							{
								for(q = 0; q < dimC; q++)
								{
									U[i * dimR + p][j * dimC + q] = localAux[p][q];
								}
							}
						}

						else
						{
							//Get the local index of the block to be processed
							for(p = 0; p < nrBlocks; p++)
							{
								//printf("\n@@@Process %d: block index %d @@@ (searched %d)", rankL, localBlockIdexes[p], globalIndexOfBlockToProcess);
								if(localBlockIdexes[p] == globalIndexOfBlockToProcess)
								{
									localIndexOfBlockToProcess = p;
									break;
								}
							}

							localAux = prodM_DD(multiplierR, dimR, dimC, localBlocks[localIndexOfBlockToProcess], dimR, dimC, 1);
							
							if (localAux == NULL)
							{
								//Error in matrix product
								free2ddouble(&multiplierR);
								free2ddouble(&multiplierC);
								free2ddouble(&localAux);

								free(localBlockIdexes);
								for(p = 0; p < nrBlocks; p++)
								{
									free2ddouble(&(localBlocks[p]));
									free2ddouble(&(localLBlocks[p]));
									free2dint(&(localPBlocks[p]));	
								}

								free(blocksPerProcess);
								MPI_Type_free(&blockType2D);
								free2dint(&processes);
								return -9;
							}
							//Sending computed results to main process (0) tag 102, + computed index
							//Sending computed index
							//printf("\nProcess %d: Sending info to process 0...", rankL);
							//MPI_Isend(&globalIndexOfBlockToProcess, 1, MPI_INT, 0, 102, MPI_COMM_WORLD, &request);
							MPI_Send(&globalIndexOfBlockToProcess, 1, MPI_INT, 0, 102, MPI_COMM_WORLD);
							//Sending U
							//MPI_Isend(&(localAux[0][0]), dimR * dimC, MPI_DOUBLE, 0, globalIndexOfBlockToProcess, MPI_COMM_WORLD, &requestU);
							MPI_Send(&(localAux[0][0]), dimR * dimC, MPI_DOUBLE, 0, globalIndexOfBlockToProcess, MPI_COMM_WORLD);	
							//MPI_Wait(&request, &status);
							//MPI_Wait(&requestU, &status);
							printf("\nProcess %d: U block sent to process 0.", rankL);
						}
						
					}
					//MPI_Wait(&request, &status);
					//free2ddouble(&multiplierR);
					//free2ddouble(&multiplierC);
					//MPI_Wait(&requestU, &status);
					//free2ddouble(&localAux);
					//MPI_Wait(&requestP, &status);
				}
			}
		}
		//MPI_Barrier(MPI_COMM_WORLD);
		if(rankL == 0)
		{	
			if(MPI_Type_vector(dimR, dimC, nrC, MPI_INT, &blockType2DINT) != 0)
			{
			//Cannot create custom MPI type
			return -7;
			}	
			MPI_Type_commit(&blockType2DINT);

			//Getting all results from each process
			for(i = 0; i < dimRP; i++)
			{
				for(j = 0; j < dimCP; j++)
				{
					int globalIndexOfBlockToProcess, r, c, **blockP;
					double **blockU, **blockL;
				
					if(malloc2ddouble(&blockU, dimR, dimC) != 0 || malloc2ddouble(&blockL, dimR, dimC) != 0 || malloc2dint(&blockP, dimR, dimC) != 0)
					{
						return -5;
					}

					if(processes[i][j] != 0)
					{
						//printf("\nReceiving info from process %d...",processes[i][j]);
						if (i == j)
						{
							//Receiving computed 2d block index
							MPI_Recv(&globalIndexOfBlockToProcess, 1, MPI_INT, processes[i][j], 100, MPI_COMM_WORLD, &status);
							//printf("\n@@@Receiving blocks %d from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							r = globalIndexOfBlockToProcess / dimCP;
							c = globalIndexOfBlockToProcess % dimCP;
							//Receiving computed block of U							
							MPI_Recv(&(U[r * dimR][c * dimC]), 1, blockType2D, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &status);
							//MPI_Recv(&(blockU[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &status); //OK		
							//printf("\n!!!!U Block %d received from process %d...", globalIndexOfBlockToProcess, processes[i][j]);	
							//printMatrixDouble(blockU, dimR, dimC);
							//Receiving computed block of L							
							MPI_Recv(&(L[r * dimR][c * dimC]), 1, blockType2D, processes[i][j], globalIndexOfBlockToProcess/10, MPI_COMM_WORLD, &status);
							//MPI_Recv(&(blockL[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][j], globalIndexOfBlockToProcess/10, MPI_COMM_WORLD, &status);//OK
							//printf("\n!!!!L Block %d received from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							//printMatrixDouble(blockL, dimR, dimC);
							//Receiving computed block of P							
							MPI_Recv(&(P[r * dimR][c * dimC]), 1, blockType2DINT, processes[i][j], globalIndexOfBlockToProcess/100, MPI_COMM_WORLD, &status);
							//MPI_Recv(&(blockP[0][0]), dimR * dimC, MPI_INT, processes[i][j], globalIndexOfBlockToProcess/100, MPI_COMM_WORLD, &status); //OK
							//printf("\n!!!!P Block %d received from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							//printMatrixInt(blockP, dimR, dimC);
						}
						else if (i > j)
						{
							//Receiving computed 2d block index from column process
							//MPI_Irecv(&globalIndexOfBlockToProcess, 1, MPI_INT, processes[i][j], 101, MPI_COMM_WORLD, &request);
							MPI_Recv(&globalIndexOfBlockToProcess, 1, MPI_INT, processes[i][j], 101, MPI_COMM_WORLD, &status);
							//printf("\n@@@ L Receiving block %d from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							r = globalIndexOfBlockToProcess / dimCP;
							c = globalIndexOfBlockToProcess % dimCP;
							//Receiving computed block of L							
							//MPI_Irecv(&(L[r * dimR][c * dimC]), 1, blockType2D, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &request);
							MPI_Recv(&(L[r * dimR][c * dimC]), 1, blockType2D, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &status);
							//MPI_Recv(&(blockL[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &status); //OK
							//MPI_Irecv(&(blockL[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &request);
							//printf("\n!!!! L Block %d received from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							//printMatrixDouble(blockL, dimR, dimC);
						}
						else
						{
							//i > j	
							//Receiving computed 2d block index from row process
							MPI_Recv(&globalIndexOfBlockToProcess, 1, MPI_INT, processes[i][j], 102, MPI_COMM_WORLD, &status);
							//MPI_Irecv(&globalIndexOfBlockToProcess, 1, MPI_INT, processes[i][j], 102, MPI_COMM_WORLD, &request);
							//printf("\n@@@ Receiving U block %d from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							r = globalIndexOfBlockToProcess / dimCP;
							c = globalIndexOfBlockToProcess % dimCP;
							//Receiving computed block of U							
							MPI_Recv(&(U[r * dimR][c * dimC]), 1, blockType2D, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &status);
							//MPI_Irecv(&(U[r * dimR][c * dimC]), 1, blockType2D, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &request);
							//MPI_Recv(&(blockU[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &status); //OK
							//MPI_Irecv(&(blockU[0][0]), dimR * dimC, MPI_DOUBLE, processes[i][j], globalIndexOfBlockToProcess, MPI_COMM_WORLD, &request); 
							//printf("\n!!!! U Block %d received from process %d...", globalIndexOfBlockToProcess, processes[i][j]);
							//printMatrixDouble(blockU, dimR, dimC);
						}
					}
				}
			}	
			stop_time = MPI_Wtime();
			printf("\nRuntime is: %f\n", stop_time - start_time);
			MPI_Type_free(&blockType2DINT);
		}
		else
		{
			int nrBlocks = blocksPerProcess[rankL];
			free(localBlockIdexes);
			for(i = 0; i < nrBlocks; i++)
			{
				free2ddouble(&(localBlocks[i]));
				free2ddouble(&(localLBlocks[i]));	
				free2dint(&(localPBlocks[i]));
			}
		}
		free(blocksPerProcess);
		MPI_Type_free(&blockType2D);
		free2dint(&processes);
		return 0;
	}
	else
	{
		return -5;
	}

}

int main(int argc, char *argv[])
{
	int nProcs, nrL, nrC, i, j, k, nrLU, nrCU, nrLL, nrCL;
	int rc, numTasks, rank, res, dimR, dimC, PR, PC;
	FILE *in;
	double **mat, **U, **L, **diagDiv;
	int **P;
	//Number of matrix lines
	nrL = atoi(argv[1]);
	//Number of matrix columns
	nrC = atoi(argv[2]);
	//Number of process grid rows
	PR = atoi(argv[3]);
	//Number of process grid columns
	PC = atoi(argv[4]);
	//Row block size
	dimR = atoi(argv[5]);
	//Column block size
	dimC = atoi(argv[6]);
	nrLU = nrL;
	nrLL = nrL;
	nrCU = nrC;
	nrCL = nrC;
	//double mat1[nrL][nrC];
	if(nrL < nrC)
	{
		nrCL = nrL;
	}
	else
	{
		nrLU = nrC;
	}
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) 
	{ 
		printf ("\n\nError starting MPI program. Terminating.\n"); 
		MPI_Abort(MPI_COMM_WORLD, rc); 
	}
	MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(malloc2ddouble(&mat, nrL, nrC) == 0 && malloc2dint(&P, nrL, nrL) == 0 && malloc2ddouble(&L, nrLL, nrCL) == 0 && malloc2ddouble(&U, nrLU, nrCU) == 0)
	{
		if(rank == 0)
		{
			in = fopen("81.txt", "r");

			for(i = 0; i < nrL; i++)
			{
				for(j = 0; j < nrC; j++)
				{	
					k = fscanf(in, "%lf",&mat[i][j]);
					if(i < nrLU && j < nrCU)
					{
						U[i][j] = mat[i][j];
					}
				}
			}

			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			printf("\nProcess %d: File closed...\nThe matrix is:\n", rank);			
			printMatrixDouble(mat, nrL, nrC);
		}
		//if(malloc2ddouble(&diagDiv, nrL, nrC) != 0)
		//{
		//	printf("\nEroare!!");
		//}	
		res = decompLU(mat, L, U, P, nrL, nrC, nrLU, nrCU, nrLL, nrCL, diagDiv);
		//res = parallelDecompLU(mat, L, U, P, nrL, nrC, nrLU, nrCU, nrLL, nrCL, numTasks, PR, PC, dimR, dimC);
		if(rank == 0)
		{
			if(res == -1)
			{
				printf("\nPivot value too small...\n");
			}
			else if(res == -2)
			{
				printf("\nMatrix is not square...\n");
			}
			else if(res == -3)
			{
				printf("\nNot sufficient resources for the specified processor grid...\n");
			}
			else if(res == -4)
			{
				printf("\nFor square matrix the processes grid should also be square...\n");
			}
			else if(res == -5)
			{
				printf("\nError in memory allocation...\n");
			}
			else if(res == -6)
			{
				printf("\nProcess grid cannot be matched with the matrix size...\n");
			}
			else if(res == -7)
			{
				printf("\nCannot create custom MPI type...\n");
			}
			else if(res == -8)
			{
				printf("\nCannot invert inferior triangular matrix...\n");
			}
			else if(res == -9)
			{
				printf("\nError in matrix product...\n");
			}
			else
			{
				printf("\nMatricea L:\n");
				printMatrixDouble(L, nrLL, nrCL);
				printf("\nMatricea U:\n");
				printMatrixDouble(U, nrLU, nrCU);
				printf("\nMatricea P:\n");
				printMatrixInt(P, nrL, nrL);
			}
		}
		

		free2ddouble(&mat);
		free2dint(&P);
		free2ddouble(&L);
		free2ddouble(&U);
	}
	else
	{
		printf("\nError in memory allocation...!\n");
	}
	MPI_Finalize();
	return 0;	
}
