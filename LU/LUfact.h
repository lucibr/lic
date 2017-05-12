#include <string.h>
#include <limits.h>

#include "matProd.h"
#include "matInv.h"

//allocMemory (input) - inddicates whether memory has been already allocated(1) for results or shoud be allocated inside the function(0)
int decompLU(double **mat, double ***L, double ***U, int ***P, int nrL, int nrC, int *nrLU, int *nrCU, int *nrLL, int *nrCL, int allocMemory, double *runtime)
{

	//Varianta neparalelizata LU = PA dreptunghiular
	
	int i, j, k, a;
	int lineMaxPivot, dim = nrL;
	double maxPivot, aux, start_time, stop_time;
	start_time = MPI_Wtime();
	if (allocMemory == 1)
	{
		if(nrL == nrC)
		{
			*nrLU = nrL;
			*nrCU = nrC;

			*nrLL = nrL;
			*nrCL = nrC;
		}	
		else if (nrL > nrC)
		{
			*nrLL = nrL;
			*nrCL = nrC;

			*nrLU = nrC;
			*nrCU = nrC;		
		}
		else
		{
			*nrLL = nrL;
			*nrCL = nrL;
		
			*nrLU = nrL;
			*nrCU = nrC;
		}
	
		if(malloc2dint(P, nrL, nrL) != 0 || malloc2ddouble(L, *nrLL, *nrCL) != 0 || malloc2ddouble(U, *nrLU, *nrCU) != 0)
		{	
			return -5;
		}	
	}

	if(nrL > nrC)
	{
		dim = nrC;
	}
	for(i = 0; i < nrL; i++)
	{
		(*P)[i][i] = 1;
	}
		
	for(i = 0; i < dim - 1; i++)
	{
		//Get the max value on the 'i' column (pivot value)
		lineMaxPivot = i;
		maxPivot = mat[i][i];
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
			printf("\n\n Pivot value %f.\n", maxPivot);
			return -1;
		}
		if(i != lineMaxPivot)
		{
			//The pivot value is not on the current line --> interchange the lines
			//Interchange lines in the original matrix
			for(j = 0; j < nrC; j++)
			{
				aux = mat[i][j];
				mat[i][j] = mat[lineMaxPivot][j];
				mat[lineMaxPivot][j] = aux;
			}
			for(j = 0; j < *nrCL; j++)
			{
				aux = (*L)[i][j];
				(*L)[i][j] = (*L)[lineMaxPivot][j];
				(*L)[lineMaxPivot][j] = aux;
			}
			for(j = 0; j < nrL; j++)
			{
				a = (*P)[i][j];
				(*P)[i][j] = (*P)[lineMaxPivot][j];
				(*P)[lineMaxPivot][j] = a;  
			}
		}
		// L coefficients (matricea multiplicatorilor)
		for(j = i; j < nrL; j++)
		{
			(*L)[j][i] = mat[j][i]/maxPivot;
		}
		// U coefficients (matricea finala)
		if(i==0)
		{
			for(j = 0; j < nrC; j++)				
			{
				(*U)[i][j] = mat[i][j];
			}		
		}
		for(j = i+1; j < nrL; j++)
			for(k = 0; k < nrC; k++)
			{
				mat[j][k] = mat[j][k] - (*L)[j][i] * mat[i][k];
				
			}

	}
	for(i = 0; i < *nrLU; i++)
	{
		for(j = 0; j < *nrCU; j++)
		{
			(*U)[i][j] = mat[i][j];
		}
	}

	if(nrL > nrC)
	{
		lineMaxPivot = nrC - 1;
		maxPivot = mat[nrC - 1][nrC - 1];
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
			//The pivot value is not on the current line --> interchange the lines
			//Interchange lines in the original matrix
			for(j = 0; j < nrC; j++)
			{
				aux = mat[nrC - 1][j];
				mat[nrC - 1][j] = mat[lineMaxPivot][j];
				mat[lineMaxPivot][j] = aux;

				aux = (*L)[nrC - 1][j];
				(*L)[nrC - 1][j] = (*L)[lineMaxPivot][j];
				(*L)[lineMaxPivot][j] = aux; 

				a = (*P)[nrC - 1][j];
				(*P)[nrC - 1][j] = (*P)[lineMaxPivot][j];
				(*P)[lineMaxPivot][nrC - 1] = a;
			}
		}
		// L coefficients (matricea multiplicatorilor)
		for(j = nrC - 1; j < nrL; j++)
		{
			(*L)[j][nrC - 1] = mat[j][nrC - 1]/maxPivot;
		}
	}
	else
	{
		(*L)[nrL-1][nrL-1] = 1;
	}
	stop_time = MPI_Wtime();
	(*runtime) = stop_time - start_time;
	return 0;
}

//nrR (INPUT) - numarul liniilor matricii
//nrC (INPUT) - numarul coloanelor matricii
//nrRU (OUTPUT) - numarul liniilor matricii triungiular superioara U 
//nrCU (OUTPUT) - numarul coloanelor matricii triungiular superioara U
//nrRL (OUTPUT) - numarul liniilor matricii triungiular inferioara L
//nrCL (OUTPUT) - numarul coloanelor matricii triungiular inferioare L
//nporcs (INPUT) - numarul proceselor/procesoarelor utilizate in calcul
//PR (INTPUT) - numarul liniilor matricii de procese
//PC (INTPUT) - numarul coloanelor matricii de procese
//dimBlock (INPUT) - dimensiunea blocului analizat de fiecare proces (matrice patratica)
int parallelDecompLU(double **mat, double ***L, double ***U, int ***P, int nrR, int nrC, int *nrRU, int *nrCU, int *nrRL, int *nrCL, int nProcs, int PR, int PC, int dimBlock, double *runtime)
{
	if(nProcs == 1)
	{
		int res;
		res = decompLU(mat, L, U, P, nrR, nrC, nrRU, nrCU, nrRL, nrCL, 1, runtime);
		return res;
	}
	else if(PC * PR > nProcs)
	{
		//Not sufficient resources for the specified processor grid
		return -3;
	}
	else if(nrR % dimBlock != 0 || nrC % dimBlock != 0)
	{
		//Process grid cannot be matched with the matrix size
		return -6;
	}
	int **processes, rankL;	
	int dimRP = nrR/dimBlock, dimCP = nrC/dimBlock; //dimensions of processes matrix	

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	if(nrR == nrC)
	{
		*nrRU = nrR;
		*nrCU = nrC;
		*nrRL = nrR;
		*nrCL = nrC;
	}
	else if (nrR > nrC)
	{
		*nrRL = nrR;
		*nrCL = nrC;

		*nrRU = nrC;
		*nrCU = nrC;		
	}
	else
	{
		*nrRL = nrR;
		*nrCL = nrR;
		
		*nrRU = nrR;
		*nrCU = nrC;
	}
	if(malloc2dint(P, nrR, nrR) != 0 || malloc2ddouble(L, *nrRL, *nrCL) != 0 || malloc2ddouble(U, *nrRU, *nrCU) != 0)
	{	
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}

	if(malloc2dint(&processes, dimRP, dimCP) == 0)
	{
		MPI_Status status;
		MPI_Request request, requestL, requestP, requestU, requestMC, requestMR;	
		double start_time, stop_time, r;
		int i, j, k, p, q;
		MPI_Datatype blockType2D, blockType2DINT;

		start_time = MPI_Wtime();

		k = 0;	
		for(p = 0; p < dimRP; p++)
		{
			for(q = 0; q < dimCP; q++)
			{
				processes[p][q] = k++;
			}
		}

		//Create MPI custom type in order to distribute to all processes 2D blocks of size dimR x dimC 
		//int MPI_Type_vector(int rowCount, int columnCount, int nrElemsToJump, MPI_Datatype oldtype, MPI_Datatype *newtype)
		if(MPI_Type_vector(dimBlock, dimBlock, nrC, MPI_DOUBLE, &blockType2D) != 0)
		{
			//Cannot create custom MPI type
			printErrorMessage(-7, rankL, "MPI_Type_vector\0");
			MPI_Abort(MPI_COMM_WORLD, -7);
			return -7;
		}
		MPI_Type_commit(&blockType2D);
		if(rankL == 0)
		{
			for(i = 0; i < dimRP; i++)
			{
				for(j = 0; j < dimCP; j++)
				{
					//Main process will not send data to itself - communication not needed since it stores the whole matrix
					p = processes[i][j];
					if(p != 0)
					{
						//Send block of dimR x dimC size
						MPI_Isend(&(mat[i * dimBlock][j * dimBlock]), 1, blockType2D, p, 0, MPI_COMM_WORLD, &request);
					}
				}
			}
		}
		int 	**localPBlock; //Stores the P values
		double	**localBlock, //Stores the actual matrix values; after processing will store U values
				**localLBlock, //Will store the L values
				dummy;
		if(rankL != 0)
		{
			if(malloc2ddouble(&(localBlock), dimBlock, dimBlock) != 0 || malloc2ddouble(&(localLBlock), dimBlock, dimBlock) != 0 || malloc2dint(&(localPBlock), dimBlock, dimBlock) != 0)
			{
				//Memory allocation error
				printErrorMessage(-5, rankL, "malloc\0");
				MPI_Abort(MPI_COMM_WORLD, -5);
				return -5;
			}
			MPI_Recv(&((localBlock)[0][0]), dimBlock*dimBlock, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);		
		}
		double	**localAux, **multiplierR, **multiplierC, **updateLblock, **updateUblock; //temporary dimR * dimC 2D array
		int **localP, res;
		for(i = 0; i < dimRP; i++)
		{
			for(j = 0; j < dimCP; j++)
			{
				if(processes[i][j] == rankL)
				{
					res = -1;

					if(malloc2ddouble(&localAux, dimBlock, dimBlock) != 0 || malloc2ddouble(&multiplierR, dimBlock, dimBlock) != 0 || malloc2ddouble(&multiplierC, dimBlock, dimBlock) != 0 || malloc2ddouble(&updateLblock, dimBlock, dimBlock) != 0 || malloc2ddouble(&updateUblock, dimBlock, dimBlock) != 0 || malloc2dint(&localP, dimBlock, dimBlock) != 0)
					{
						//Memory allocation error
						printErrorMessage(-5, rankL, "malloc\0");
						MPI_Abort(MPI_COMM_WORLD, -5);
						return -5;
					}
					if(i == j)
					{
						if(rankL != 0)
						{
							if (i != 0)
							{
								for (p = 0; p < i; p++)
								{
									MPI_Recv(&(updateLblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD, &status);
									MPI_Recv(&(updateUblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD, &status);
									localAux = prodM_DD(updateLblock, dimBlock, dimBlock, updateUblock, dimBlock, dimBlock);
									sumM_DD(localBlock, localAux, 1, -1, dimBlock, dimBlock, &r, &(localBlock));
								}
							}				
							
							res = decompLU(localBlock, &(localLBlock), &localAux, &(localPBlock), dimBlock, dimBlock, &dimBlock, &dimBlock, &dimBlock, &dimBlock, 0, &dummy);

							if(res != 0)
							{
								printErrorMessage(res, rankL, "decompLU\0");
								MPI_Abort(MPI_COMM_WORLD, res);
								return res;
							}
							
							if(supMatInv(localAux, multiplierC, dimBlock) != 0)
							{
								//Cannot invert superior triangular matrix
								printErrorMessage(-8, rankL, "supMatInv\0");
								MPI_Abort(MPI_COMM_WORLD, -8);
								return -8;
							}
							//Send to the multiplier to all inferior column processes
							for(p = i+1; p < dimCP; p++)
							{
								MPI_Isend(&(multiplierC[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 2, MPI_COMM_WORLD, &requestMC);
							}
							//Computing multiplier for row processes Linv * P
							//Computing Linv
							if(infMatInv(localLBlock, localAux, dimBlock) != 0)
							{
								//Cannot invert inferior triangular matrix
								printErrorMessage(-8, rankL, "infMatInv\0");
								MPI_Abort(MPI_COMM_WORLD, -8);
								return -8;
							}

							//Computing Linv * P
							multiplierR = prodM_DI(localAux, dimBlock, dimBlock, localPBlock, dimBlock, dimBlock);
							if(multiplierR == NULL)
							{
								//Error in matrix product	
								printErrorMessage(-9, rankL, "prodM_DI\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}

							//Send multiplier and P to row processes
							for(p = 0; p < dimCP; p++)
							{
								if(p > j)
								{
									MPI_Isend(&(multiplierR[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 3, MPI_COMM_WORLD, &requestMR);
								}
								else if(p < j)
								{
									MPI_Isend(&((localPBlock)[0][0]), dimBlock * dimBlock, MPI_INT, processes[i][p], 4, MPI_COMM_WORLD, &requestMR);
								}
							}
							
							//Sending computed results to main process (0) tag 100, + computed index
							//Sending U
							MPI_Isend(&((localBlock)[0][0]), dimBlock * dimBlock, MPI_DOUBLE, 0, 333, MPI_COMM_WORLD, &requestU);
							//Sending L
							MPI_Isend(&((localLBlock)[0][0]), dimBlock * dimBlock, MPI_DOUBLE, 0, 3333, MPI_COMM_WORLD, &requestL);
							//Sending P
							MPI_Isend(&((localPBlock)[0][0]), dimBlock * dimBlock, MPI_INT, 0, 33333, MPI_COMM_WORLD, &requestP);
						}


						else
						{
							//Process 0 sholud process data taken directly from matrix mat 					
							double	**localA, **localL;
							if(malloc2ddouble(&localA, dimBlock, dimBlock) != 0 || malloc2ddouble(&localL, dimBlock, dimBlock) != 0)
							{

								printErrorMessage(-5, rankL, "malloc\0");
								MPI_Abort(MPI_COMM_WORLD, -5);
								return -5;
							}
							for(p = 0; p < dimBlock; p++)
							{
								for(q = 0; q < dimBlock; q++)
								{
									localA[p][q] = mat[i * dimBlock + p][j * dimBlock + q];
								}
							}
							if (j != 0)
							{
								for (p = 0; p < j; p++)
								{
									MPI_Recv(&(updateLblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD, &status);
									MPI_Recv(&(updateUblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD, &status);
									localAux = prodM_DD(updateLblock, dimBlock, dimBlock, updateUblock, dimBlock, dimBlock);
									sumM_DD(localA, localAux, 1, -1, dimBlock, dimBlock, &r, &(localA));
								}
							}
							
							res = decompLU(localA, &localL, &localAux, &localP, dimBlock, dimBlock, &dimBlock, &dimBlock, &dimBlock, &dimBlock, 0, &dummy);

							if(res != 0)
							{

								printErrorMessage(res, rankL, "decompLU\0");
								MPI_Abort(MPI_COMM_WORLD, res);
								return res;
							}

							//Copy results into position in the final result containers L, U, P
							for(p = 0; p < dimBlock; p++)
							{
								for(q = 0; q < dimBlock; q++)
								{
									(*L)[i * dimBlock + p][j * dimBlock + q] = localL[p][q];
									(*U)[i * dimBlock + p][j * dimBlock + q] = localAux[p][q];
									(*P)[i * dimBlock + p][j * dimBlock + q] = localP[p][q];
								}
							}
							
							if(supMatInv(localAux, multiplierC, dimBlock) != 0)
							{
								//Cannot invert superior triangular matrix
								printErrorMessage(-8, rankL, "supMatInv\0");
								MPI_Abort(MPI_COMM_WORLD, -8);
								return -8;
							}
							//Send to the multiplier to all inferior column processes
							for(p = i+1; p < dimCP; p++)
							{
								MPI_Isend(&(multiplierC[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 2, MPI_COMM_WORLD, &requestMC);
							}
							
							//Computing multiplier for row processes Linv * P
							//Computing Linv
							if(infMatInv(localL, localAux, dimBlock) != 0)
							{
								//Cannot invert inferior triangular matrix

								printErrorMessage(-8, rankL, "infMatInv\0");
								MPI_Abort(MPI_COMM_WORLD, -8);
								return -8;
							}
							//Computing Linv * P
							multiplierR = prodM_DI(localAux, dimBlock, dimBlock, localP, dimBlock, dimBlock);
							if(multiplierR == NULL)
							{
								//Error in matrix product

								printErrorMessage(-9, rankL, "prodM_DI\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}

							//Send multiplier and P to row processes
							for(p = 0; p < dimCP; p++)
							{
								if(p > j)
								{
									MPI_Isend(&(multiplierR[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 3, MPI_COMM_WORLD, &requestMR);
								}
								else if(p < j)
								{
									MPI_Send(&(localP[0][0]), dimBlock * dimBlock, MPI_INT, processes[i][p], 4, MPI_COMM_WORLD);
								}
							}
						}
					}

					else if(i > j)
					{
						//This is a column process
						MPI_Recv(&(multiplierC[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[j][j], 2, MPI_COMM_WORLD, &status);

						if (rankL == 0)
						{
							//Process 0 sholud process data taken directly from matrix mat 					
							double	**localA;
							if(malloc2ddouble(&localA, dimBlock, dimBlock) != 0)
							{
								printErrorMessage(-5, rankL, "malloc\0");
								MPI_Abort(MPI_COMM_WORLD, -5);
								return -5;
							}
							//Copy the block that will be processed
							for(p = 0; p < dimBlock; p++)
							{
								for(q = 0; q < dimBlock; q++)
								{
									localA[p][q] = mat[i * dimBlock + p][j * dimBlock + q];
								}
							}

							if (j != 0)
							{
								for (p = 0; p < j; p++)
								{
									MPI_Recv(&(updateLblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD, &status);
									MPI_Recv(&(updateUblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD, &status);
									localAux = prodM_DD(updateLblock, dimBlock, dimBlock, updateUblock, dimBlock, dimBlock);
									sumM_DD(localA, localAux, 1, -1, dimBlock, dimBlock, &r, &(localA));
									
								}
							}
							localAux = prodM_DD(localA, dimBlock, dimBlock, multiplierC, dimBlock, dimBlock);

							if (localAux == NULL)
							{
								//Error in matrix product
								printErrorMessage(-9, rankL, "prodM_DD\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}
							//Send L updateblocks to all processes, to the right, on row i
							for(p = j+1; p < dimCP; p++)
							{
								MPI_Send(&(localAux[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD);
							}
							//Receiving P matrix from process on the same line, on the main diagonal
							MPI_Recv(&(localP[0][0]), dimBlock * dimBlock, MPI_INT, processes[i][i], 4, MPI_COMM_WORLD, &status);
						
							localA = prodM_ID(localP, dimBlock, dimBlock, localAux, dimBlock, dimBlock);

							if (localA == NULL)
							{
								//Error in matrix product
								printErrorMessage(-9, rankL, "prodM_ID\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}
							//Copy results into position in the final result container L
							for(p = 0; p < dimBlock; p++)
							{
								for(q = 0; q < dimBlock; q++)
								{
									(*L)[i * dimBlock + p][j * dimBlock + q] = localA[p][q];
								}
							}
						}

						else
						{	
							if (j != 0)
							{
								for (p = 0; p < j; p++)
								{
									MPI_Recv(&(updateLblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD, &status);
									MPI_Recv(&(updateUblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD, &status);
									localAux = prodM_DD(updateLblock, dimBlock, dimBlock, updateUblock, dimBlock, dimBlock);
									sumM_DD(localBlock, localAux, 1, -1, dimBlock, dimBlock, &r, &(localBlock));
								}
							}
							localAux = prodM_DD(localBlock, dimBlock, dimBlock, multiplierC, dimBlock, dimBlock);
							if (localAux == NULL)
							{
								//Error in matrix product
								printErrorMessage(-9, rankL, "prodM_DD\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}
						
							//Send L updateblocks to all processes, to the right, on row i
							for(p = j+1; p < dimCP; p++)
							{
								MPI_Send(&(localAux[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD);
							}
							//Receiving P matrix from process on the same line, on the main diagonal
							MPI_Recv(&(localP[0][0]), dimBlock * dimBlock, MPI_INT, processes[i][i], 4, MPI_COMM_WORLD, &status);

							localAux = prodM_ID(localP, dimBlock, dimBlock, localAux, dimBlock, dimBlock);

							if (localAux == NULL)
							{
								//Error in matrix product
								printErrorMessage(-9, rankL, "prodM_ID\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}
							//Sending computed results to main process (0) tag 101, + computed index
							//Sending L
							MPI_Send(&(localAux[0][0]), dimBlock * dimBlock, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);	
						}		
					}
					else
					{
						//i < j -> This is a row process
						MPI_Recv(&(multiplierR[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][i], 3, MPI_COMM_WORLD, &status);

						if (rankL == 0)
						{
							//Process 0 sholud process data taken directly from matrix mat 					
							double	**localA;
							if(malloc2ddouble(&localA, dimBlock, dimBlock) != 0)
							{
								printErrorMessage(-5, rankL, "malloc\0");
								MPI_Abort(MPI_COMM_WORLD, -5);
								return -5;
							}
							//Copy the block that will be processed
						
							for(p = 0; p < dimBlock; p++)
							{
								for(q = 0; q < dimBlock; q++)
								{
									localA[p][q] = mat[i * dimBlock + p][j * dimBlock + q];
								}
							}

							if (i != 0)
							{
								for (p = 0; p < i; p++)
								{
									MPI_Recv(&(updateLblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD, &status);
									MPI_Recv(&(updateUblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD, &status);
									localAux = prodM_DD(updateLblock, dimBlock, dimBlock, updateUblock, dimBlock, dimBlock);
									sumM_DD(localBlock, localAux, 1, -1, dimBlock, dimBlock, &r, &(localBlock));
								}
							}
						
							localAux = prodM_DD(multiplierR, dimBlock, dimBlock, localA, dimBlock, dimBlock);

							if (localAux == NULL)
							{
								//Error in matrix product
								printErrorMessage(-9, rankL, "prodM_DD\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}
							//Send U updateblocks to all processes on column j
							for(p = i+1; p < dimBlock; p++)
							{
								MPI_Send(&(localAux[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD);
							}
							//Copy results into position in the final result container L
							for(p = 0; p < dimBlock; p++)
							{
								for(q = 0; q < dimBlock; q++)
								{
									(*U)[i * dimBlock + p][j * dimBlock + q] = localAux[p][q];
								}
							}
						}

						else
						{
							if (i != 0)
							{
								for (p = 0; p < i; p++)
								{
									MPI_Recv(&(updateLblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[i][p], 11, MPI_COMM_WORLD, &status);
									MPI_Recv(&(updateUblock[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD, &status);
									localAux = prodM_DD(updateLblock, dimBlock, dimBlock, updateUblock, dimBlock, dimBlock);
									sumM_DD(localBlock, localAux, 1, -1, dimBlock, dimBlock, &r, &(localBlock));
								}
							}
						
							localAux = prodM_DD(multiplierR, dimBlock, dimBlock, localBlock, dimBlock, dimBlock);
						
							if (localAux == NULL)
							{
								//Error in matrix product
								printErrorMessage(-9, rankL, "prodM_DD\0");
								MPI_Abort(MPI_COMM_WORLD, -9);
								return -9;
							}
							//Send U updateblocks to all processes on column j
							for(p = i+1; p < dimRP; p++)
							{
								MPI_Send(&(localAux[0][0]), dimBlock * dimBlock, MPI_DOUBLE, processes[p][j], 13, MPI_COMM_WORLD);
							}
							//Sending computed results to main process (0) tag 102, + computed index
							//Sending U
							MPI_Send(&(localAux[0][0]), dimBlock * dimBlock, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);	
						}
					}
				}
			}
		}
		if(rankL == 0)
		{	
			if(MPI_Type_vector(dimBlock, dimBlock, nrC, MPI_INT, &blockType2DINT) != 0)
			{
				//Cannot create custom MPI type
				printErrorMessage(-7, rankL, "MPI_Type_vector\0");
				MPI_Abort(MPI_COMM_WORLD, -7);
				return -7;
			}	
			MPI_Type_commit(&blockType2DINT);
			//Getting all results from each process
			for(i = 0; i < dimRP; i++)
			{
				for(j = 0; j < dimCP; j++)
				{
					int **blockP;
					double **blockU, **blockL;
				
					if(malloc2ddouble(&blockU, dimBlock, dimBlock) != 0 || malloc2ddouble(&blockL, dimBlock, dimBlock) != 0 || malloc2dint(&blockP, dimBlock, dimBlock) != 0)
					{
						printErrorMessage(-5, rankL, "malloc\0");
						MPI_Abort(MPI_COMM_WORLD, -5);
						return -5;
					}

					if(processes[i][j] != 0)
					{
						if (i == j)
						{
							//Receiving computed block of U							
							MPI_Recv(&((*U)[i * dimBlock][j * dimBlock]), 1, blockType2D, processes[i][j], 333, MPI_COMM_WORLD, &status);
							//Receiving computed block of L							
							MPI_Recv(&((*L)[i * dimBlock][j * dimBlock]), 1, blockType2D, processes[i][j], 3333, MPI_COMM_WORLD, &status);
							//Receiving computed block of P							
							MPI_Recv(&((*P)[i * dimBlock][j * dimBlock]), 1, blockType2DINT, processes[i][j], 33333, MPI_COMM_WORLD, &status);
						}
						else if (i > j)
						{
							//Receiving computed block of L							
							MPI_Recv(&((*L)[i * dimBlock][j * dimBlock]), 1, blockType2D, processes[i][j], 101, MPI_COMM_WORLD, &status);
						}
						else
						{
							//i > j
							//Receiving computed block of U							
							MPI_Recv(&((*U)[i * dimBlock][j * dimBlock]), 1, blockType2D, processes[i][j], 102, MPI_COMM_WORLD, &status);
						}
					}
				}
			}	
			stop_time = MPI_Wtime();
			*runtime = stop_time - start_time;
			MPI_Type_free(&blockType2DINT);
		}
		MPI_Type_free(&blockType2D);
		free2dint(&processes);
		return 0;
	}
	else
	{
		printErrorMessage(-5, rankL, "malloc\0");
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}

}
