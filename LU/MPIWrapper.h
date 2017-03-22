#include "mpi.h"

//Initializes MPI framework
void MPI_Framework_Init(int argc, char *argv[], int *numberOfProcesses)
{
	int ok;
	ok = MPI_Init(&argc, &argv);
	if (ok != MPI_SUCCESS)
	{
		printf("\nMPI framework could not be initialized.");
		MPI_Abort(MPI_COMM_WORLD, ok);
	}
	else
	{
		MPI_Comm_size(MPI_COMM_WORLD, numberOfProcesses);
	}
}

void MPI_Framework_Stop()
{
	MPI_Finalize();
}
