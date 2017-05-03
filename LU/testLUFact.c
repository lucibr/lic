#include "LUfact.h"

int main(int argc, char *argv[])
{
	int nrL, nrC, i, j, k, nrLU, nrCU, nrLL, nrCL;
	int numTasks, rank, res, dimBlock, PR, PC;
	FILE *in;
	double **mat, **U, **L, runtime;
	char *filename;
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

	filename = argv[6];

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
			in = fopen(filename, "r");

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
			//printMatrixDouble(mat, nrL, nrC);
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
				//printf("\nMatricea L:\n");
				//printMatrixDouble(L, nrLL, nrCL);
				//printf("\nMatricea U:\n");
				//printMatrixDouble(U, nrLU, nrCU);
				//printf("\nMatricea P:\n");
			    //printMatrixInt(P, nrL, nrL);
				printf("\n\nTimpul de executie: %f\n", runtime);
			}
		}

/*		res = sumM_DD_P(L, U, 1, 1, nrL, nrC, 1, &runtime, &L);*/

/*		if (res == 0 && rank == 0)*/
/*			{*/
/*				printf("\nMatricea U+L:\n");*/
/*				printMatrixDouble(L, nrL, nrC);*/
/*				printf("\n\nTimpul de executie: %f\n", runtime);*/
/*			}*/
/*		free2ddouble(&mat);	*/
	}
	else
	{
		printf("\nError in memory allocation...!\n");
	}
	MPI_Framework_Stop();
	return 0;	
}

