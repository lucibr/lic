#include "matSum.h"

double** prodM_DI(double **mat1, int nrL1, int nrC1, int **mat2, int nrL2, int nrC2)
{
	int i, j, k;
	double **resultMat;
	if(nrC1 != nrL2)
	{
		printErrorMessage(-10, 0, "parallelProdM_DD_D\0");
		return NULL;
	}

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
	return NULL;
}

double** prodM_ID(int **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2)
{
	double **resultMat;
	int i, j, k;
	if(nrC1 != nrL2)
	{
		printErrorMessage(-10, 0, "parallelProdM_DD_D\0");
		return NULL;
	}

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
	return NULL;
}

double** prodM_DD(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2)
{
	double **resultMat;
	int i, j, k;
	if(nrC1 != nrL2)
	{
		printErrorMessage(-10, 0, "parallelProdM_DD_D\0");
		return NULL;
	}
	
	if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
	{
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
		return resultMat;		
	}
	else
	{
		//Memory could not be allocated
		return NULL;
	}
	return NULL;
}

double** parallelProdM_DD_D(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, int nProcs)
{
	//The Output Data decomposition method is applied
	int *ranks, i, j, k, rankL, nrElem, nrSupElem, maxNumberOfLines, 
		*toGet, //will store the number of elements from the result matrix to be obtained from each process
		*offsets, //will store the offsets where the elements from each process will be storred (offsets of **resultMat)
		*recevedLinesNumbers, //Will store the numbers of lines from the first matrix to be sent to each process
		*receivedLineOffsets; //will store the first matrix start offsets of lines to be sent to each process

    double *receivedLineElements, //Will store the first matrix elements sent to each process
     	   **resultMat, //will store the product AxB result
		   *resultElem; //will store the elements computed by each process
	//Total number of matrix elements
	unsigned long n;

	//Number of elements in the result matrix
	n = nrL1 * nrC2;
	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	if(nProcs > n)
	{
		if(rankL == 0)
			printf("\nToo many processes specified: %lu elements have to be computed by %d processes. Aborting...\n", n, nProcs); 
		return NULL;
	}
	else if(nrC1 != nrL2)
	{
		if(rankL == 0)
			printErrorMessage(-10, rankL, "parallelProdM_DD_D\0");
		return NULL;
	}
	else
	{
		//Array will keep the index of processes from the current group which will be added in a new group
		ranks = (int *) malloc(sizeof(int) * nProcs);
		for(i = 0; i < nProcs; i++)
		{
			ranks[i] = i;
		}
		if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
		{
			//The parallel code begins here	
			MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
			
			//Array will keep the number of elements to be computed by each process
			toGet = (int *) malloc(sizeof(int) * nProcs);
			//Array will keep the offsets relative to elements container address
			offsets = (int *) malloc(sizeof(int) * nProcs);

			nrElem = n/nProcs;	
			nrSupElem = n%nProcs;
			
			//Defining offsets and number of elements to be sent to each process (information is available in every process)	
			for(i = 0; i < nProcs; i++)
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
			resultElem = (double *) malloc(sizeof(double) * toGet[rankL]);
			//Alocating space to store the number of elements from first matrix to be sent to each process
			recevedLinesNumbers = (int *) malloc(sizeof(int) * nProcs);

			//Alocating space for first matrix line offsets
			receivedLineOffsets = (int *) malloc(sizeof(int) * nProcs);
			maxNumberOfLines = 1;
			//Lines indexes (offsets) and number of elements of matrix A to be sent to each process
			for(i = 0; i < nProcs; i++)
			{
				int minLineIndex, maxLineIndex;
				minLineIndex = offsets[i] / nrC2;
				if(i == 0)
				{	
					maxLineIndex = (offsets[i + 1] - 1)/ nrC2;
					recevedLinesNumbers[i] = maxLineIndex;
					receivedLineOffsets[i] = minLineIndex * nrC1;
				}
				if(i == nProcs - 1)
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
			receivedLineElements = (double *) malloc(sizeof(double) * maxNumberOfLines * nrC1);
			//Broadcasting second matrix
			MPI_Bcast(&(mat2[0][0]), nrL2*nrC2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Scatterv(&(mat1[0][0]), recevedLinesNumbers, receivedLineOffsets, MPI_DOUBLE, receivedLineElements, maxNumberOfLines * nrC1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
			}
			MPI_Gatherv(resultElem, toGet[rankL], MPI_DOUBLE, &(resultMat[0][0]), toGet, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			free(toGet);
			free(offsets);
			free(resultElem);
			free(recevedLinesNumbers);
			free(receivedLineElements);
			free(receivedLineOffsets);
			free(ranks);					
			return resultMat;
		}
		else
		{
			//Memory could not be allocated
			return NULL;
		}
	}
}

//Implementarea variantei de inmultire paralela a matricilor prin calcul sistolic
int parallelProdM_DD_SIST(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, double *runtime, double ***result, int nProcs, int dimBlockL1, int dimBlockC1, int dimBlockL2, int dimBlockC2)
{
	int rankL, i, j, nrLB, nrCB, aux, send = 1, receive = 1,
		**procMatrix; //will store the process rank that will compute the associated product block

	MPI_Status status;

	MPI_Datatype blocK2DM1, //sub-block first matrix
				 blocK2DM2, //sub-block second matrix
				 blocK2DR; //sub-block result matrix

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	
	if(nrC1 != nrL2)
	{
		return -10;
	}
	//Check if the sub-block dimensions can be used
	// First condition: the actual matrix dimensions should be divisible with the sub-block dimensions
	if(nrL1 % dimBlockL1 != 0 || nrC1 % dimBlockC1 != 0 || nrL2 % dimBlockL2 != 0 || nrC2 % dimBlockC2 != 0)
	{
		return -6;
	}
	//Check if the sub-block dimensions can be used
	// Second condition: the number of blocks should be the same for both, matrix A and matrix B
	if(nrL1 / dimBlockL1 != nrL2 / dimBlockL2 || nrC1 / dimBlockC1 != nrC2 / dimBlockC2)
	{
		return -6;
	}
	//Check if the sub-block dimensions can be used
	// Third condition: sufficient processes
	if((nrL1 / dimBlockL1) * (nrL2 / dimBlockL2) + 1 > nProcs)
	{
		return -3;
	}
	nrLB = nrL1/dimBlockL1;
	nrCB = nrC1/dimBlockC1;
	//nrLB should be equal to nrCB
	if(malloc2dint(&procMatrix, nrLB, nrCB) != 0)
	{	
		if(rankL == 0)
		{
			printErrorMessage(-5, rankL, "parallelProdM_DD_SIST\0");
		}
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}
	//Fill up the process matrix - process 0 (main process excluded)
	aux = 1;
	for(i = 0; i < nrLB; i++)
	{
		for(j = 0; j < nrCB; j++)
		{
			procMatrix[i][j] = aux++;
		}
	}

 	if(rankL == 0)
	{
		//int MPI_Type_vector(int rowCount, int columnCount, int nrElemsToJump, MPI_Datatype oldtype, MPI_Datatype *newtype)
		//Create MPI custom type for the first matrix sub-block
		if(MPI_Type_vector(dimBlockL1, dimBlockC1, nrC1, MPI_DOUBLE, &blocK2DM1) != 0)
		{
			//Cannot create custom MPI type
			printErrorMessage(-7, rankL, "MPI_Type_vector - parallelProdM_DD_SIST\0");
			MPI_Abort(MPI_COMM_WORLD, -7);
			return -7;
		}
		MPI_Type_commit(&blocK2DM1);

		//Create MPI custom type for the second matrix sub-block
		if(MPI_Type_vector(dimBlockL2, dimBlockC2, nrC2, MPI_DOUBLE, &blocK2DM2) != 0)
		{
			//Cannot create custom MPI type
			printErrorMessage(-7, rankL, "MPI_Type_vector - parallelProdM_DD_SIST\0");
			MPI_Abort(MPI_COMM_WORLD, -7);
			return -7;
		}
		MPI_Type_commit(&blocK2DM2);

		//The main process will send matrix 1 sub blocks (a column at once) to first column processes and send matrix 2 sub blocks (a row at once) to first row processes; 
		//The sending process will start with last matrix 1 column respectvely with the last matrix 2 row;
		//Each process will receive 0 as a sign that the all data was sent;
				
		for(i = nrLB - 1; i >= 0; i--)
		{
			//Sending pairs: column sub-block from matrix 1 to column process / row sub-block from matrix 2 to row process
			for(j = 0; j < nrLB; j++)
			{
				//Sending continue tag				
				MPI_Send(&send, 1, MPI_INT, procMatrix[j][0], 1, MPI_COMM_WORLD);
				
				//Send block of dimBlockL1 x dimBlockC1 size (line j, column i)
				MPI_Send(&(mat1[j * dimBlockL1][i * dimBlockC1]), 1, blocK2DM1, procMatrix[j][0], 0, MPI_COMM_WORLD);

				//Sending continue tag				
				MPI_Send(&send, 1, MPI_INT, procMatrix[0][j], 1, MPI_COMM_WORLD);

				//Send block of dimBlockL1 x dimBlockC1 size (line i, column j)
				MPI_Send(&(mat2[i * dimBlockL2][j * dimBlockC2]), 1, blocK2DM2, procMatrix[0][j], 0, MPI_COMM_WORLD);
			}
		}
		//Send stop signal
		send = 0;
		for(j = 0; j < nrLB; j++)
		{
			MPI_Send(&send, 1, MPI_INT, procMatrix[j][0], 1, MPI_COMM_WORLD);
			MPI_Send(&send, 1, MPI_INT, procMatrix[0][j], 1, MPI_COMM_WORLD);
		}
	}
	else
	{
		double **aBlock, **bBlock, **resBlock, **auxBlock, dummyAux;
		if(malloc2ddouble(&aBlock, dimBlockL1, dimBlockC1) != 0 || malloc2ddouble(&bBlock, dimBlockL2, dimBlockC2) != 0 || malloc2ddouble(&resBlock, dimBlockL1, dimBlockC2) != 0)
		{
			if(rankL == 0)
			{
				printErrorMessage(-5, rankL, "parallelProdM_DD_SIST\0");
			}
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}

		for(i = 0; i < nrLB; i++)
		{
			for(j = 0; j < nrCB; j++)
			{
				if(procMatrix[i][j] == rankL)
				{
					while(receive == 1)
					{
						if(i == 0 && j == 0)
						{
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
							if(receive == 1)
							{
								//Receive A block (from left)
								MPI_Recv(&(aBlock[0][0]), dimBlockL1 * dimBlockC1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
							}
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
							if(receive == 1)
							{
								//Receive A block (from top)
								MPI_Recv(&(bBlock[0][0]), dimBlockL2 * dimBlockC2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
							}
							send = receive;

							//Send aBlock to the right process
							//Sending continue tag				
							MPI_Send(&send, 1, MPI_INT, procMatrix[i][j+1], 1, MPI_COMM_WORLD);
							if(send == 1)
							{
								//Send block of dimBlockL1 x dimBlockC1 size (line j, column i)
								MPI_Send(&(aBlock[0][0]), dimBlockL1*dimBlockC1, MPI_DOUBLE, procMatrix[i][j+1], 0, MPI_COMM_WORLD);
							}

							//Send bBlock to the bottom process
							MPI_Send(&send, 1, MPI_INT, procMatrix[i+1][j], 1, MPI_COMM_WORLD);
							if(send == 1)
							{
								//Send block of dimBlockL2 x dimBlockC2 size (line j, column i)
								MPI_Send(&(bBlock[0][0]), dimBlockL2*dimBlockC2, MPI_DOUBLE, procMatrix[i+1][j], 0, MPI_COMM_WORLD);
							}
							if(receive == 1)
							{
								auxBlock = prodM_DD(aBlock, dimBlockL1, dimBlockC1, bBlock, dimBlockL2, dimBlockC2);
								if(auxBlock == NULL)
								{
									printErrorMessage(-10, rankL, "parallelProdM_DD_SIST\0");
									MPI_Abort(MPI_COMM_WORLD, -10);
									return -10;
								}
								//sumM_DD(double **mat1, double **mat2, double f1, double f2, int nrL, int nrC, double *runtime, double ***result)
								if(sumM_DD(resBlock, auxBlock, 1, 1, dimBlockL1, dimBlockC2, &dummyAux, &resBlock) != 0)
								{
									printErrorMessage(-100, rankL, "parallelProdM_DD_SIST/sumM_DD\0");
									MPI_Abort(MPI_COMM_WORLD, -100);
									return -100;
								}
							}
						}
						else if(i == 0)
						{
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, procMatrix[i][j-1], 1, MPI_COMM_WORLD, &status);
							if(receive == 1)
							{
								//Receive A block (from left)
								MPI_Recv(&(aBlock[0][0]), dimBlockL1 * dimBlockC1, MPI_DOUBLE, procMatrix[i][j-1], 0, MPI_COMM_WORLD, &status);
							}
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
							if(receive == 1)
							{
								//Receive A block (from top)
								MPI_Recv(&(bBlock[0][0]), dimBlockL2 * dimBlockC2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
							}
							send = receive;

							//Send aBlock to the right process (if this is not the last process on the current process row)
							if(j+1 < nrCB)
							{
								//Sending continue tag				
								MPI_Send(&send, 1, MPI_INT, procMatrix[i][j+1], 1, MPI_COMM_WORLD);
								if(send == 1)
								{
									//Send block of dimBlockL1 x dimBlockC1 size (line j, column i)
									MPI_Send(&(aBlock[0][0]), dimBlockL1*dimBlockC1, MPI_DOUBLE, procMatrix[i][j+1], 0, MPI_COMM_WORLD);
								}
							}

							//Send bBlock to the bottom process
							MPI_Send(&send, 1, MPI_INT, procMatrix[i+1][j], 1, MPI_COMM_WORLD);
							if(send == 1)
							{
								//Send block of dimBlockL2 x dimBlockC2 size (line j, column i)
								MPI_Send(&(bBlock[0][0]), dimBlockL2*dimBlockC2, MPI_DOUBLE, procMatrix[i+1][j], 0, MPI_COMM_WORLD);
							}
							if(receive == 1)
							{
								auxBlock = prodM_DD(aBlock, dimBlockL1, dimBlockC1, bBlock, dimBlockL2, dimBlockC2);
								if(auxBlock == NULL)
								{
									printErrorMessage(-10, rankL, "parallelProdM_DD_SIST\0");
									MPI_Abort(MPI_COMM_WORLD, -10);
									return -10;
								}
								//sumM_DD(double **mat1, double **mat2, double f1, double f2, int nrL, int nrC, double *runtime, double ***result)
								if(sumM_DD(resBlock, auxBlock, 1, 1, dimBlockL1, dimBlockC2, &dummyAux, &resBlock) != 0)
								{
									printErrorMessage(-100, rankL, "parallelProdM_DD_SIST/sumM_DD\0");
									MPI_Abort(MPI_COMM_WORLD, -100);
									return -100;
								}
							}
						}
						else if(j == 0)
						{
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
							if(receive == 1)
							{
								//Receive A block (from left)
								MPI_Recv(&(aBlock[0][0]), dimBlockL1 * dimBlockC1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
							}
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, procMatrix[i-1][j], 1, MPI_COMM_WORLD, &status);	
							if(receive == 1)
							{
								//Receive A block (from top)
								MPI_Recv(&(bBlock[0][0]), dimBlockL2 * dimBlockC2, MPI_DOUBLE, procMatrix[i-1][j], 0, MPI_COMM_WORLD, &status);
							}
							send = receive;

							//Send aBlock to the right process
							//Sending continue tag				
							MPI_Send(&send, 1, MPI_INT, procMatrix[i][j+1], 1, MPI_COMM_WORLD);
							if(send == 1)
							{
								//Send block of dimBlockL1 x dimBlockC1 size (line j, column i)
								MPI_Send(&(aBlock[0][0]), dimBlockL1*dimBlockC1, MPI_DOUBLE, procMatrix[i][j+1], 0, MPI_COMM_WORLD);
							}

							//Send bBlock to the bottom process (if the current process is not already the last on the current process column)
							if(i+1 < nrLB)
							{
								MPI_Send(&send, 1, MPI_INT, procMatrix[i+1][j], 1, MPI_COMM_WORLD);
								if(send == 1)
								{
									//Send block of dimBlockL2 x dimBlockC2 size (line j, column i)
									MPI_Send(&(bBlock[0][0]), dimBlockL2*dimBlockC2, MPI_DOUBLE, procMatrix[i+1][j], 0, MPI_COMM_WORLD);
								}
							}
							if(receive == 1)
							{
								auxBlock = prodM_DD(aBlock, dimBlockL1, dimBlockC1, bBlock, dimBlockL2, dimBlockC2);
								if(auxBlock == NULL)
								{
									printErrorMessage(-10, rankL, "parallelProdM_DD_SIST\0");
									MPI_Abort(MPI_COMM_WORLD, -10);
									return -10;
								}
								//sumM_DD(double **mat1, double **mat2, double f1, double f2, int nrL, int nrC, double *runtime, double ***result)
								if(sumM_DD(resBlock, auxBlock, 1, 1, dimBlockL1, dimBlockC2, &dummyAux, &resBlock) != 0)
								{
									printErrorMessage(-100, rankL, "parallelProdM_DD_SIST/sumM_DD\0");
									MPI_Abort(MPI_COMM_WORLD, -100);
									return -100;
								}
							}
						}
						else
						{
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, procMatrix[i][j-1], 1, MPI_COMM_WORLD, &status);
							if(receive == 1)
							{
								//Receive A block (from left)
								MPI_Recv(&(aBlock[0][0]), dimBlockL1 * dimBlockC1, MPI_DOUBLE, procMatrix[i][j-1], 0, MPI_COMM_WORLD, &status);
							}
							//Receive continue signal
							MPI_Recv(&receive, 1, MPI_INT, procMatrix[i-1][j], 1, MPI_COMM_WORLD, &status);
							
							if(receive == 1)
							{
								//Receive A block (from top)
								MPI_Recv(&(bBlock[0][0]), dimBlockL2 * dimBlockC2, MPI_DOUBLE, procMatrix[i-1][j], 0, MPI_COMM_WORLD, &status);
							}
							send = receive;

							//Send aBlock to the right process (if the current process is not already the last process on the current process row)
							if(j+1 < nrCB)
							{
								//Sending continue tag				
								MPI_Send(&send, 1, MPI_INT, procMatrix[i][j+1], 1, MPI_COMM_WORLD);
								if(send == 1)
								{
									//Send block of dimBlockL1 x dimBlockC1 size (line j, column i)
									MPI_Send(&(aBlock[0][0]), dimBlockL1*dimBlockC1, MPI_DOUBLE, procMatrix[i][j+1], 0, MPI_COMM_WORLD);
								}
							}

							//Send bBlock to the bottom process (if the current process is not already the last on the current process column)
							if(i+1 < nrLB)
							{
								MPI_Send(&send, 1, MPI_INT, procMatrix[i+1][j], 1, MPI_COMM_WORLD);
								if(send == 1)
								{
									//Send block of dimBlockL2 x dimBlockC2 size (line j, column i)
									MPI_Send(&(bBlock[0][0]), dimBlockL2*dimBlockC2, MPI_DOUBLE, procMatrix[i+1][j], 0, MPI_COMM_WORLD);
								}
							}
							if(receive == 1)
							{
								auxBlock = prodM_DD(aBlock, dimBlockL1, dimBlockC1, bBlock, dimBlockL2, dimBlockC2);
								if(auxBlock == NULL)
								{
									printErrorMessage(-10, rankL, "parallelProdM_DD_SIST\0");
									MPI_Abort(MPI_COMM_WORLD, -10);
									return -10;
								}
								//sumM_DD(double **mat1, double **mat2, double f1, double f2, int nrL, int nrC, double *runtime, double ***result)
								if(sumM_DD(resBlock, auxBlock, 1, 1, dimBlockL1, dimBlockC2, &dummyAux, &resBlock) != 0)
								{
									printErrorMessage(-100, rankL, "parallelProdM_DD_SIST/sumM_DD\0");
									MPI_Abort(MPI_COMM_WORLD, -100);
									return -100;
								}
							}
						}
					}
					MPI_Send(&(resBlock[0][0]), dimBlockL1*dimBlockC2, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD);
					return 0;
				}
			}
		}

	}
	
	//Receive the processed info into result matrix
	if(rankL == 0)
	{
		if(MPI_Type_vector(dimBlockL1, dimBlockC2, nrC2, MPI_DOUBLE, &blocK2DR) != 0)
		{
			//Cannot create custom MPI type
			printErrorMessage(-7, rankL, "MPI_Type_vector - parallelProdM_DD_SIST\0");
			MPI_Abort(MPI_COMM_WORLD, -7);
			return -7;
		}
		MPI_Type_commit(&blocK2DR);

		for(i = 0; i < nrLB; i++)
		{
			for(j = 0; j < nrCB; j++)
			{
				MPI_Recv(&((*result)[i * dimBlockL1][j * dimBlockC2]), 1, blocK2DR, procMatrix[i][j], 1000, MPI_COMM_WORLD, &status);
			}
		}
		free2ddouble(&aBlock);
		free2ddouble(&bBlock);
		free2ddouble(&resBlock);
		MPI_Type_free(&blocK2DR);
		MPI_Type_free(&blocK2DM1);
		MPI_Type_free(&blocK2DM2);
		return 0;
	}
	return 0;
}

int parallelProdM_DD(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, double *runtime, double ***result, int nProcs, int dimBlockL1, int dimBlockC1, int dimBlockL2, int dimBlockC2)
{
	int rankL, res;
	double start_time, stop_time;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	
	if(nrC1 != nrL2)
	{
		printErrorMessage(-10, rankL, "parallelProdM_DD\0");
		return -10;
	}
	else if(malloc2ddouble(result, nrL1, nrC2) != 0)
	{	
		printErrorMessage(-5, rankL, "parallelProdM_DD\0");
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}
	else if(nProcs == 1)
	{
		start_time = MPI_Wtime();
		*result =  prodM_DD(mat1, nrL1, nrC1, mat2, nrL2, nrC2);
		if(*result == NULL)
		{
			printErrorMessage(-100, rankL, "parallelProdM_DD\0");
			return -100;
		}
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return 0;
	}
	else
	{
		start_time = MPI_Wtime();
		res = parallelProdM_DD_SIST(mat1, nrL1, nrC1, mat2, nrL2, nrC2, runtime, result, nProcs, dimBlockL1, dimBlockC1, dimBlockL2, dimBlockC2);
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		if(res == 0)
		{
			return 0;
		}
		else
		{
			start_time = MPI_Wtime();
			*result =  parallelProdM_DD_D(mat1, nrL1, nrC1, mat2, nrL2, nrC2, nProcs);
			if(*result == NULL)
			{
				printErrorMessage(-100, rankL, "parallelProdM_DD\0");
				return -100;
			}
			stop_time = MPI_Wtime();
			*runtime = stop_time - start_time;
			return 0;
		}
	}
}

int prodMV_DD_S(double** mat, int nrL, int nrC, double *v, int dim, double **result)
{
	int i, j;
	if(nrC != dim)
		return -10;
	*result = (double *)calloc(nrL, sizeof(double));
	if(!result)
		return -5;
	for(i = 0; i < nrL; i++)
	{
		for(j = 0; j < nrC; j++)
		{
			(*result)[i] += (mat[i][j]*v[j]);
		}
	}
	return 0;
}

//Pipeline implementation
int parallelProdMV_DD(double **mat, int nrL, int nrC, double *v, int dim, double **result, double *runtime, int nProcs)
{
	int i, j, rankL, nrElem, nrSupElem, 
		*offsets, //will store the offsets from where elements of the vector line will be sent to the pipeline processes
		*toGet; //will store the number of elements to be  sent to each process

	double  start_time, stop_time, localSum,
			*receivedVectorElements, //will store the vector elements
			*receivedMatrixElements; //will store the matrix line elements (will change a number of times equal with the number of matrix lines)

	//The parallel code begins here
	start_time = MPI_Wtime();

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);

	//Array will keep the number of elements to be computed by each process
	toGet = (int *) malloc(sizeof(int) * nProcs);
	//Array will keep the offsets relative to elements container address
	offsets = (int *) malloc(sizeof(int) * nProcs);

	nrElem = nrC / (nProcs - 1);	
	nrSupElem = nrC % (nProcs - 1);
	//Defining offsets and number of elements to be sent to each process (information is available in every process)	
	for(i = 0; i < nProcs; i++)
	{
		//For a distribution as balanced as possible, for the first processes a supplementary elem is considered
		//Normaly, each process will process elements
		if(i == 0)
		{
			toGet[i] = 0;
		}
		else if(i <= nrSupElem)
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
		if(i == 0 || i == 1)
		{
			offsets[i] = 0;
		}
		else if(i <= nrSupElem )
		{
			offsets[i] = (i*nrElem+nrSupElem-i);
		}
		else
		{
			offsets[i] = (i*nrElem+nrSupElem)-1;
		}
	}
	//Alocating space to store the vector elements 
	receivedVectorElements = (double *) malloc(sizeof(double) * toGet[rankL]);
	//Alocating space to store the matrix line elements 
	receivedMatrixElements = (double *) malloc(sizeof(double) * toGet[rankL]);

	if(rankL == 0)
	{
		*result = (double *)calloc(nrL, sizeof(double));
		if(!result)
		{
			printErrorMessage(-5, rankL, "parallelProdMV_DD\0");
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
	}

	if(rankL != 0)
	{
		if(malloc2ddouble(&mat, nrL, 1) != 0)
		{
			printErrorMessage(-5, rankL, "parallelProdMV_DD\0");
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
	}

	//Scattring the vector elements to all processes except the main process (rank 0)
	MPI_Scatterv(&(v[0]), toGet, offsets, MPI_DOUBLE, receivedVectorElements, toGet[rankL], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for(i = 0; i < nrL; i++)
	{
		//Scattring the matrix elements to all processes except the main process (rank 0)
		MPI_Scatterv(&(mat[i][0]), toGet, offsets, MPI_DOUBLE, receivedMatrixElements, toGet[rankL], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		localSum = 0;
		if(rankL != 0)
		{			
			for(j = 0; j < toGet[rankL]; j++)
			{			
				localSum += (receivedVectorElements[j] * receivedMatrixElements[j]);
			}
		}
		MPI_Reduce(&localSum, &((*result)[i]), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	stop_time = MPI_Wtime();
	*runtime = stop_time - start_time;
	return 0;
}

int prodMV_DD(double** mat, int nrL, int nrC, double *v, int dim, double **result, double *runtime, int nProcs)
{
	int rankL, res;
	double start_time, stop_time;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	if(nrC != dim)
	{
		return -10;
	}
	if(nProcs == 1 || nProcs == 2)
	{
		if(rankL == 0)
		{
			start_time = MPI_Wtime();
			res = prodMV_DD_S(mat, nrL, nrC, v, dim, result);
			stop_time = MPI_Wtime();
			*runtime = stop_time - start_time;
		}
		return res;
	}
	else
	{
		return parallelProdMV_DD(mat, nrL, nrC, v, dim, result, runtime, nProcs);
	}
}
