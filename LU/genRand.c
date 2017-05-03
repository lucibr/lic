#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
	int i, nr, upperBound;
	FILE *out;
	double aux;
	char *filename;
	//filename
	filename = argv[1];
	nr = atoi(argv[2]);
	upperBound = atoi(argv[3]);

	srand(time(NULL));

	out = fopen(filename, "w");

	for(i = 0; i < nr; i++)
	{
		aux = rand() % (upperBound + 1 - (-1)*upperBound) - upperBound;
		fprintf(out,"%lf ", aux);
	}
	fclose(out);
	printf("\nFile %s generated and closed...\n", filename);
	return 0;	
}
