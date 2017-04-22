#include <stdio.h>

#define EPS 0.01  

void printErrorMessage(int errorCode, int MPI_Process_Rank, char* functionName)
{
	switch(errorCode)
	{
		case -1:
			printf("\nProcess %d (function %s): Pivot value too small...\n", MPI_Process_Rank, functionName);
			break;
		case -2:
			printf("\nProcess %d (function %s): Sub-block dimension should be at least 3...\n", MPI_Process_Rank, functionName);
			break;
		case -3:
			printf("\nProcess %d (function %s): Not sufficient resources for the specified processor grid...\n", MPI_Process_Rank, functionName);
			break;
		case -4:
			printf("\nProcess %d (function %s): For square matrix the processes grid should also be square...\n", MPI_Process_Rank, functionName);
			break;
		case -5:
			printf("\nProcess %d (function %s): Error in memory allocation...\n", MPI_Process_Rank, functionName);
			break;
		case -6:
			printf("\nProcess %d (function %s): Process grid cannot be matched with the matrix size...\n", MPI_Process_Rank, functionName);
			break;
		case -7:
			printf("\nProcess %d (function %s): Cannot create custom MPI type...\n", MPI_Process_Rank, functionName);
			break;
		case -8:
			printf("\nProcess %d (function %s): Cannot invert triangular matrix...\n", MPI_Process_Rank, functionName);
			break;
		case -9:
			printf("\nProcess %d (function %s): Error in matrix product...\n", MPI_Process_Rank, functionName);
			break;
		case -10:
			printf("\nProcess %d (function %s): The matrices cannot be multiplied! (nrC(A) != nrL(B))...\n", MPI_Process_Rank, functionName);
			break;
		case -11:
			printf("\nProcess %d (function %s): Cannot create MPI group...\n", MPI_Process_Rank, functionName);
			break;
		case -12:
			printf("\nProcess %d (function %s): Cannot create MPI communicator...\n", MPI_Process_Rank, functionName);
			break;
		case -13:
			printf("\nProcess %d (function %s): Incorrectly defined system...\n", MPI_Process_Rank, functionName);
			break;
		default:
			printf("\nProcess %d (function %s): Unknown error...\n", MPI_Process_Rank, functionName);
			break;
	}
}
