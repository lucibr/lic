
#include "matProd.h"

int main(int argc, char *argv[])
{
	int i, j, k, nrL1, nrC1, nrL2, nrC2;
	int numTasks, rank, status, dimBlockL1, dimBlockC1, dimBlockL2, dimBlockC2;
	FILE *in;
	double **mat1, **mat2, runtime, **res;
	char *filename;
	//Number of matrix 1 lines
	nrL1 = atoi(argv[1]);
	//Number of matrix 1 columns
	nrC1 = atoi(argv[2]);
	//Number of matrix 2 lines
	nrL2 = atoi(argv[3]);
	//Number of matrix 2 columns
	nrC2 = atoi(argv[4]);
	//Matrix A Sub-block: number of lines
	dimBlockL1 = atoi(argv[5]);
	//Matrix A Sub-block: number of columns
	dimBlockC1 = atoi(argv[6]);
	//Matrix B Sub-block: number of lines
	dimBlockL2 = atoi(argv[7]);
	//Matrix B Sub-block: number of columns
	dimBlockC2 = atoi(argv[8]);

	filename = argv[9];

	MPI_Framework_Init(argc, argv, &numTasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(malloc2ddouble(&mat1, nrL1, nrC1) == 0 && malloc2ddouble(&mat2, nrL2, nrC2) == 0)
	{
		if(rank == 0)
		{
			in = fopen(filename, "r");

			for(i = 0; i < nrL1; i++)
			{
				for(j = 0; j < nrC1; j++)
				{	
					k = fscanf(in, "%lf",&mat1[i][j]);
				}
			}
			printf("\nProcess %d: The matrix 1 is:\n", rank);		
			printMatrixDouble(mat1, nrL1, nrC1);


			for(i = 0; i < nrL2; i++)
			{
				for(j = 0; j < nrC2; j++)
				{	
					k = fscanf(in, "%lf",&mat2[i][j]);
				}
			}
			printf("\nProcess %d: The matrix 2 is:\n", rank);			
			printMatrixDouble(mat2, nrL2, nrC2);
			printf("\nProcess %d: Reading done...Closing file...", rank);
			fclose(in);
		}
		status = parallelProdM_DD(mat1, nrL1, nrC1, mat2, nrL2, nrC2, &runtime, &res, numTasks, dimBlockL1, dimBlockC1, dimBlockL2, dimBlockC2);
		if(rank == 0)
		{
			if(status == 0)
			{
					printf("\nMatricea rezultat este:\n");
					printMatrixDouble(res, nrL1, nrC2);
					printf("\n\nTimpul de executie: %f (number of processes: %d)\n", runtime, numTasks);
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

