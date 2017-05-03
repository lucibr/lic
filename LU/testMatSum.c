#include "matSum.h"

int main(int argc, char *argv[])
{
	int i, j, k, nrL, nrC;
	int numTasks, rank, status;
	FILE *in;
	double **mat1, **mat2, runtime, **res, f1, f2;
	//Number of matrix 1 lines
	nrL = atoi(argv[1]);
	//Number of matrix 1 columns
	nrC = atoi(argv[2]);
	//Matrix 1 factor
	f1 = atof(argv[3]);
	//Matrix 2 factor
	f2 = atoi(argv[4]);

	MPI_Framework_Init(argc, argv, &numTasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(malloc2ddouble(&mat1, nrL, nrC) == 0 && malloc2ddouble(&mat2, nrL, nrC) == 0)
	{
		if(rank == 0)
		{
			in = fopen("81.txt", "r");

			for(i = 0; i < nrL; i++)
			{
				for(j = 0; j < nrC; j++)
				{	
					k = fscanf(in, "%lf",&mat1[i][j]);
				}
			}
			//printf("\nProcess %d: The matrix 1 is:\n", rank);		
			//printMatrixDouble(mat1, nrL, nrC);


			for(i = 0; i < nrL; i++)
			{
				for(j = 0; j < nrC; j++)
				{	
					k = fscanf(in, "%lf",&mat2[i][j]);
				}
			}
			//printf("\nProcess %d: The matrix 2 is:\n", rank);			
			//printMatrixDouble(mat2, nrL, nrC);
			printf("\nProcess %d: Reading done...Closing file...", rank);
			fclose(in);
		}
		status = sumM_DD_P(mat1, mat2, f1, f2, nrL, nrC, numTasks, &runtime, res);
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0)
		{
			if(status == 0)
			{
					//printf("\nMatricea rezultat este:\n");
					//printMatrixDouble(*res, nrL, nrC);
					printf("\n\nTimpul de executie: %f (Numar procese: %d)\n", runtime, numTasks);
			}
			else
			{
				printErrorMessage(status, rank, "main\0");
			}
		}
	}
	MPI_Framework_Stop();
	return 0;	
}

