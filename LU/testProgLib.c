#include <dlfcn.h>
#include "matMPIOp.h"

int main(int argc, char *argv[])
 {
	int nrL, nrC, dim, i, j, numTasks, rank, k, status;
	double **mat, *v, runtime, *result;
	FILE *in;
	//Number of matrix 1 lines
	nrL = atoi(argv[1]);
	//Number of matrix 1 columns
	nrC = atoi(argv[2]);
	//Number of matrix 2 lines
	dim = atoi(argv[3]);

	char *error;
	void *handle;

	handle = dlopen("libmatMPIOp.so", RTLD_LAZY);
	if(!handle)
	{
		fputs(dlerror(), stderr);
		exit(1);
	}

/*	MPI_Framework_Init = dlsym(handle, "MPI_Framework_Init");*/
/*	if( (error = dlerror()) != NULL)*/
/*	{*/
/*		fputs(error, stderr);*/
/*		exit(1);*/
/*	}*/

/*	MPI_Framework_Stop = dlsym(handle, "MPI_Framework_Stop");*/
/*	if( (error = dlerror()) != NULL)*/
/*	{*/
/*		fputs(error, stderr);*/
/*		exit(1);*/
/*	}*/

	malloc2ddouble = dlsym(handle, "malloc2ddouble");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	printErrorMessage = dlsym(handle, "printErrorMessage");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	printMatrixDouble = dlsym(handle, "printMatrixDouble");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	printVectorDouble = dlsym(handle, "printVectorDouble");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}

	prodMV_DD = dlsym(handle, "prodMV_DD");
	if( (error = dlerror()) != NULL)
	{
		fputs(error, stderr);
		exit(1);
	}
		
	MPI_Framework_Init(argc, argv, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(rank == 0)
	{
		if(malloc2ddouble(&mat, nrL, nrC) != 0)
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

		for(i = 0; i < nrL; i++)
		{
			for(j = 0; j < nrC; j++)
			{	
				k = fscanf(in, "%lf",&mat[i][j]);
			}
		}
		printf("\nProcess %d: The matrix is:\n", rank);		
		printMatrixDouble(mat, nrL, nrC);

		for(i = 0; i < dim; i++)
		{
			k = fscanf(in, "%lf",&v[i]);
		}
		printf("\nProcess %d: The vector is:\n", rank);			
		printVectorDouble(v, dim);
		printf("\nProcess %d: Reading done...Closing file...\n", rank);
		fclose(in);
	}
	status = prodMV_DD(mat, nrL, nrC, v, dim, &result, &runtime, numTasks);
	if(rank == 0)
	{
		if(status == 0)
		{
				printf("\nVectorul rezultat este:\n");
				printVectorDouble(result, nrL);
				printf("\n\nTimpul de executie: %f\n", runtime);
		}
		else
		{
			printErrorMessage(status, rank, "main\0");
		}
	}
	MPI_Framework_Stop();
	return 0;	
}

