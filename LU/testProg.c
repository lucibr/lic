#include "gaussElim.h"

int main(int argc, char *argv[])
{

	
	int nrL, matDim, dim, i, j, numTasks, rank, k, status, blockDim;
	double **mat, *v, runtime, *result;
	FILE *in;
	matDim = atoi(argv[1]);
	dim = atoi(argv[2]);
	blockDim = atoi(argv[3]);

	
	MPI_Framework_Init(argc, argv, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0)
	{
		if(malloc2ddouble(&mat, matDim, matDim) != 0)
		{
			printErrorMessage(-5, rank, "main\0");
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
		v = (double *)calloc(dim, sizeof(double));
		if(!v)
		{
			printErrorMessage(-5, rank, "main\0");
			MPI_Abort(MPI_COMM_WORLD, -5);
			return -5;
		}
	
		in = fopen("81.txt", "r");

		for(i = 0; i < matDim; i++)
		{
			for(j = i; j < matDim; j++)
			{	
				k = fscanf(in, "%lf",&mat[i][j]);
			}
		}
		printf("\nProcess %d: The matrix is:\n", rank);		
		printMatrixDouble(mat, matDim, matDim);

		for(i = 0; i < dim; i++)
		{
			k = fscanf(in, "%lf",&v[i]);
		}
		printf("\nProcess %d: The vector is:\n", rank);			
		printVectorDouble(v, dim);
		printf("\nProcess %d: Reading done...Closing file...\n", rank);
		fclose(in);
	}
	status = backSubst(mat, matDim, v, dim, &result, blockDim, &runtime, numTasks);
	if(rank == 0)
	{
		if(status == 0)
		{
				printf("\nVectorul rezultat este:\n");
				printVectorDouble(result, matDim);
				printf("\n\nTimpul de executie: %f\n", runtime);
		}
		else
		{
			printErrorMessage(status, rank, "main\0");
		}
	}

	MPI_Framework_Stop();
	return 0;


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
