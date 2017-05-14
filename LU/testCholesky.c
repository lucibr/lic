#include "CHOLESKYfact.h"

int main(int argc, char *argv[])
{
	int matDim, i, j, k, numTasks, rank, res, dimBlock;
	FILE *in;
	double **mat, **L, runtime;
	char *filename;
	//Matrix dimensions
	matDim = atoi(argv[1]);
	//Block dimension
	dimBlock = atoi(argv[2]);

	filename = argv[3];

	MPI_Framework_Init(argc, argv, &numTasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(malloc2ddouble(&mat, matDim, matDim) == 0)
	{
		if(rank == 0)
		{
			in = fopen(filename, "r");

			for(i = 0; i < matDim; i++)
			{
				for(j = 0; j <= i; j++)
				{	
					k = fscanf(in, "%lf",&mat[i][j]);
					mat[j][i] = mat[i][j];
				}
			}
			printf("\nProcess %d: Reading done...", rank);
			fclose(in);
			//printf("\nProcess %d: File closed...\nThe matrix is:\n", rank);			
			//printMatrixDouble(mat, matDim, matDim);
		}
		res = choleskyFact(mat, matDim, &L, dimBlock, &runtime, numTasks);
		if(rank == 0)
		{
			if (res == 0)
			{
				//printf("\nMatricea L:\n");
				//printMatrixDouble(L, matDim, matDim);
				printf("\n\nTimpul de executie: %f\n", runtime);
			}
			else
			{
				printErrorMessage(res, rank, "choleskyFact\0");
			}
		}
	}
	else
	{
		printf("\nError in memory allocation...!\n");
	}
	MPI_Framework_Stop();
	return 0;	
}
