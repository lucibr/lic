double** prodM_DI(double **mat1, int nrL1, int nrC1, int **mat2, int nrL2, int nrC2)
{
	int i, j, k;
	double **resultMat;
	if(nrC1 != nrL2)
	{
		printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
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
		printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
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
		printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
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
			printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
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
			//printf("\nProcess %d: Broadcasting...\n", rankL);
			//Broadcasting second matrix
			MPI_Bcast(&(mat2[0][0]), nrL2*nrC2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			//printf("\nProcess %d: Broadcast done!\n", rankL);

			//printf("\nProcess %d: Scattering...\n", rankL);
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
				//printf("\n(Process %d):  elem%d%d computed %d", rankL, line_index, column_index, elem);
			}

			//printf("\nProcess %d - \nLocal operations finished...\n", rank);
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

int parallelProdM_DD_SIST(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, double *runtime, double ***result, int nProcs)
{
	return 0;
}

int parallelProdM_DD(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2, double *runtime, double ***result, int nProcs)
{
	int rankL;
	double start_time, stop_time;

	MPI_Comm_rank(MPI_COMM_WORLD, &rankL);
	
	if(nrC1 != nrL2)
	{
		return -10;
	}
	else if(malloc2ddouble(result, nrL1, nrC2) != 0)
	{	
		MPI_Abort(MPI_COMM_WORLD, -5);
		return -5;
	}
	else if(nProcs == 1)
	{
		start_time = MPI_Wtime();
		*result =  prodM_DD(mat1, nrL1, nrC1, mat2, nrL2, nrC2);
		if(*result == NULL)
		{
			return -100;
		}
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return 0;
	}
	else
	{
		start_time = MPI_Wtime();
		*result =  parallelProdM_DD_D(mat1, nrL1, nrC1, mat2, nrL2, nrC2, nProcs);
		if(*result == NULL)
		{
			return -100;
		}
		stop_time = MPI_Wtime();
		*runtime = stop_time - start_time;
		return 0;
	}
	return 0;
}
