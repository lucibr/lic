#include "LUfact.h"
int main(int argc, char *argv[])
{


	//----------------------------- LU main ------------------------------------------
	int nrL, nrC, i, j, k, nrLU, nrCU, nrLL, nrCL;
	int numTasks, rank, res, dimBlock, PR, PC;
	FILE *in;
	double **mat, **U, **L, runtime;
	int **P;
	//Number of matrix lines
	nrL = atoi(argv[1]);
	//Number of matrix columns
	nrC = atoi(argv[2]);
	//Number of process grid rows
	PR = atoi(argv[3]);
	//Number of process grid columns
	PC = atoi(argv[4]);
	//Sub-block size
	dimBlock = atoi(argv[5]);

	nrLU = nrL;
	nrLL = nrL;	
	nrCU = nrC;
	nrCL = nrC;

	if(nrL < nrC)
	{
		nrCL = nrL;
	}
	else
	{
		nrLU = nrC;
	}
	MPI_Framework_Init(argc, argv, &numTasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(malloc2ddouble(&mat, nrL, nrC) == 0)
	{
		if(rank == 0)
		{
			in = fopen("81.txt", "r");

			for(i = 0; i < nrL; i++)
			{
				for(j = 0; j < nrC; j++)
				{	
					k = fscanf(in, "%lf",&mat[i][j]);
				}
			}
			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			printf("\nProcess %d: File closed...\nThe matrix is:\n", rank);			
			printMatrixDouble(mat, nrL, nrC);
		}
		res = parallelDecompLU(mat, &L, &U, &P, nrL, nrC, &nrLU, &nrCU, &nrLL, &nrCL, numTasks, PR, PC, dimBlock, &runtime);
		if(res < 0)
		{
			printErrorMessage(res, rank, "parallelDecompLU\0");
		}
		if(rank == 0)
		{
			if (res == 0)
			{
				printf("\nMatricea L:\n");
				printMatrixDouble(L, nrLL, nrCL);
				printf("\nMatricea U:\n");
				printMatrixDouble(U, nrLU, nrCU);
				printf("\nMatricea P:\n");
			    printMatrixInt(P, nrL, nrL);
				printf("\n\nTimpul de executie: %f\n", runtime);
			}
		}

		res = sumM_DD_P(L, U, 1, 1, nrL, nrC, 1, &runtime, &L);

		if (res == 0 && rank == 0)
			{
				printf("\nMatricea U+L:\n");
				printMatrixDouble(L, nrL, nrC);
				printf("\n\nTimpul de executie: %f\n", runtime);
			}
		free2ddouble(&mat);	
	}
	else
	{
		printf("\nError in memory allocation...!\n");
	}
	MPI_Framework_Stop();
	return 0;	
	//----------------------------- LU main ------------------------------------------

/*	int nrL, matDim, i, j, numTasks, rank, k, status, blockDim;*/
/*	double **mat, runtime, **result;*/
/*	FILE *in;*/
/*	matDim = atoi(argv[1]);*/
/*	blockDim = atoi(argv[2]);*/

/*	*/
/*	MPI_Framework_Init(argc, argv, &numTasks);*/
/*	MPI_Comm_rank(MPI_COMM_WORLD, &rank);*/

/*	if(rank == 0)*/
/*	{*/
/*		if(malloc2ddouble(&mat, matDim, matDim) != 0)*/
/*		{*/
/*			printErrorMessage(-5, rank, "main\0");*/
/*			MPI_Abort(MPI_COMM_WORLD, -5);*/
/*			return -5;*/
/*		}*/
/*	*/
/*		in = fopen("ch4x4.txt", "r");*/

/*		for(i = 0; i < matDim; i++)*/
/*		{*/
/*			for(j = 0; j < matDim; j++)*/
/*			{	*/
/*				k = fscanf(in, "%lf",&mat[i][j]);*/
/*			}*/
/*		}*/
/*		printf("\nProcess %d: The matrix is:\n", rank);		*/
/*		printMatrixDouble(mat, matDim, matDim);*/
/*		printf("\nProcess %d: Reading done...Closing file...\n", rank);*/
/*		fclose(in);*/
/*	}*/
/*	status = choleskyFact(mat, matDim, &result, blockDim, &runtime, numTasks);*/
/*	if(rank == 0)*/
/*	{*/
/*		if(status == 0)*/
/*		{*/
/*				//printf("\nVectorul rezultat este:\n");*/
/*				//printMatrixDouble(result, matDim, matDim);*/
/*				//printf("\n\nTimpul de executie: %f\n", runtime);*/
/*		}*/
/*		else*/
/*		{*/
/*			printErrorMessage(status, rank, "main\0");*/
/*		}*/
/*	}*/

/*	MPI_Framework_Stop();*/
/*	return 0;*/


//------------------------------------------------------------------------------------BACK SUBSTITUTION MAIN-----------------------------------------------------------------	
/*	int nrL, matDim, dim, i, j, numTasks, rank, k, status, blockDim;*/
/*	double **mat, *v, runtime, *result;*/
/*	FILE *in;*/
/*	matDim = atoi(argv[1]);*/
/*	dim = atoi(argv[2]);*/
/*	blockDim = atoi(argv[3]);*/

/*	*/
/*	MPI_Framework_Init(argc, argv, &numTasks);*/
/*	MPI_Comm_rank(MPI_COMM_WORLD, &rank);*/

/*	if(rank == 0)*/
/*	{*/
/*		if(malloc2ddouble(&mat, matDim, matDim) != 0)*/
/*		{*/
/*			printErrorMessage(-5, rank, "main\0");*/
/*			MPI_Abort(MPI_COMM_WORLD, -5);*/
/*			return -5;*/
/*		}*/
/*		v = (double *)calloc(dim, sizeof(double));*/
/*		if(!v)*/
/*		{*/
/*			printErrorMessage(-5, rank, "main\0");*/
/*			MPI_Abort(MPI_COMM_WORLD, -5);*/
/*			return -5;*/
/*		}*/
/*	*/
/*		in = fopen("81.txt", "r");*/

/*		for(i = 0; i < matDim; i++)*/
/*		{*/
/*			for(j = i; j < matDim; j++)*/
/*			{	*/
/*				k = fscanf(in, "%lf",&mat[i][j]);*/
/*			}*/
/*		}*/
/*		printf("\nProcess %d: The matrix is:\n", rank);		*/
/*		printMatrixDouble(mat, matDim, matDim);*/

/*		for(i = 0; i < dim; i++)*/
/*		{*/
/*			k = fscanf(in, "%lf",&v[i]);*/
/*		}*/
/*		printf("\nProcess %d: The vector is:\n", rank);			*/
/*		printVectorDouble(v, dim);*/
/*		printf("\nProcess %d: Reading done...Closing file...\n", rank);*/
/*		fclose(in);*/
/*	}*/
/*	status = backSubst(mat, matDim, v, dim, &result, blockDim, &runtime, numTasks);*/
/*	if(rank == 0)*/
/*	{*/
/*		if(status == 0)*/
/*		{*/
/*				printf("\nVectorul rezultat este:\n");*/
/*				printVectorDouble(result, matDim);*/
/*				printf("\n\nTimpul de executie: %f\n", runtime);*/
/*		}*/
/*		else*/
/*		{*/
/*			printErrorMessage(status, rank, "main\0");*/
/*		}*/
/*	}*/

/*	MPI_Framework_Stop();*/
/*	return 0;*/
//------------------------------------------------------------------------------------BACK SUBSTITUTION MAIN-----------------------------------------------------------------	

	//------------------------------------------------------------------------------------FORWARD SUBSTITUTION MAIN-----------------------------------------------------------------
/*	int nrL, matDim, dim, i, j, numTasks, rank, k, status, blockDim;*/
/*	double **mat, *v, runtime, *result;*/
/*	FILE *in;*/
/*	matDim = atoi(argv[1]);*/
/*	dim = atoi(argv[2]);*/
/*	blockDim = atoi(argv[3]);*/

/*	*/
/*	MPI_Framework_Init(argc, argv, &numTasks);*/
/*	MPI_Comm_rank(MPI_COMM_WORLD, &rank);*/

/*	if(rank == 0)*/
/*	{*/
/*		if(malloc2ddouble(&mat, matDim, matDim) != 0)*/
/*		{*/
/*			printErrorMessage(-5, rank, "main\0");*/
/*			MPI_Abort(MPI_COMM_WORLD, -5);*/
/*			return -5;*/
/*		}*/
/*		v = (double *)calloc(dim, sizeof(double));*/
/*		if(!v)*/
/*		{*/
/*			printErrorMessage(-5, rank, "main\0");*/
/*			MPI_Abort(MPI_COMM_WORLD, -5);*/
/*			return -5;*/
/*		}*/
/*	*/
/*		in = fopen("81.txt", "r");*/

/*		for(i = 0; i < matDim; i++)*/
/*		{*/
/*			for(j = 0; j < i+1; j++)*/
/*			{	*/
/*				k = fscanf(in, "%lf",&mat[i][j]);*/
/*			}*/
/*		}*/
/*		printf("\nProcess %d: The matrix is:\n", rank);		*/
/*		printMatrixDouble(mat, matDim, matDim);*/

/*		for(i = 0; i < dim; i++)*/
/*		{*/
/*			k = fscanf(in, "%lf",&v[i]);*/
/*		}*/
/*		printf("\nProcess %d: The vector is:\n", rank);			*/
/*		printVectorDouble(v, dim);*/
/*		printf("\nProcess %d: Reading done...Closing file...\n", rank);*/
/*		fclose(in);*/
/*	}*/
/*	status = forwardSubst(mat, matDim, v, dim, &result, blockDim, &runtime, numTasks);*/
/*	if(rank == 0)*/
/*	{*/
/*		if(status == 0)*/
/*		{*/
/*				printf("\nVectorul rezultat este:\n");*/
/*				printVectorDouble(result, matDim);*/
/*				printf("\n\nTimpul de executie: %f\n", runtime);*/
/*		}*/
/*		else*/
/*		{*/
/*			printErrorMessage(status, rank, "main\0");*/
/*		}*/
/*	}*/

/*	MPI_Framework_Stop();*/
/*	return 0;	*/
	//------------------------------------------------------------------------------------FORWARD SUBSTITUTION MAIN-----------------------------------------------------------------
}
