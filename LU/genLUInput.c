#include <stdlib.h>
#include <string.h>

#include "matProd.h"

int main(int argc, char *argv[])
{
	int i, j, k, nrL1, nrC1, nrL2, nrC2, p;
	int numTasks, rank, status, dimBlockL1, dimBlockC1, dimBlockL2, dimBlockC2;
	FILE *inL, *inU, *out, *outE;
	double **mat1, **mat2, runtime, **res;
	char filename[20], filenameE[25];
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

	MPI_Framework_Init(argc, argv, &numTasks);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(malloc2ddouble(&mat1, nrL1, nrC1) == 0 && malloc2ddouble(&mat2, nrL2, nrC2) == 0)
	{
		if(rank == 0)
		{
			inU = fopen("81.txt", "r");

			for(i = 0; i < nrL1; i++)
			{
				for(j = i; j < nrC1; j++)
				{	
					k = fscanf(inU, "%lf",&mat2[i][j]);
				}
			}
			printf("\nProcess %d: Reading done...Closing file...", rank);
			fclose(inU);
			printf("\nProcess %d: The matrix 2 is:\n", rank);		
			printMatrixDouble(mat2, nrL2, nrC2);
			
			inL = fopen("rand01.txt", "r");

			for(i = 0; i < nrL2; i++)
			{
				for(j = 0; j <= i; j++)
				{	
					if(i == j)
					{
						mat1[i][j] = 1;
					}
					else
					{
						k = fscanf(inL, "%lf",&mat1[i][j]);
					}
				}
			}
			printf("\nProcess %d: Reading done...Closing file...", rank);
			fclose(inL);
			printf("\nProcess %d: The matrix 1 is:\n", rank);			
			printMatrixDouble(mat1, nrL1, nrC1);
		}
		status = parallelProdM_DD(mat1, nrL1, nrC1, mat2, nrL2, nrC2, &runtime, &res, numTasks, dimBlockL1, dimBlockC1, dimBlockL2, dimBlockC2);
		if(rank == 0)
		{
			if(status == 0)
			{
					printf("\nMatricea rezultat este:\n");
					printMatrixDouble(res, nrL1, nrC2);
					printf("\n\nTimpul de executie: %f (number of processes: %d)\n", runtime, numTasks);
					
					snprintf(filename, 20,"LUinput%d.txt",nrL1);
					snprintf(filenameE, 25,"LUinput2x%d.txt",nrL1);

					out = fopen(filename, "w");
										

					for(i = 0; i < nrL1; i++)
					{
						for(j = 0; j < nrC2; j++)
						{
							fprintf(out, "%f ", res[i][j]);
						}
						fprintf(out, "\n");
					}
					fclose(out);

					outE = fopen(filenameE, "w");
					for(p = 0; p <= 1; p++)
					{
						for(i = 0; i < nrL1; i++)
						{
							for(k = 0; k <= 1; k++)
							{
								for(j = 0; j < nrC2; j++)
								{
									fprintf(out, "%f ", res[i][j]);
								}
							}
							fprintf(outE,"\n");
						}
					}			
					fclose(outE);		
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

