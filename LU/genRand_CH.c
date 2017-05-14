#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
	int i, nr, upperBoundDIAG, upperBound, j;
	FILE *out;
	double aux;
	char *filename;
	//filename
	filename = argv[1];
	nr = atoi(argv[2]);
	upperBoundDIAG = atoi(argv[3]);
	upperBound = atoi(argv[4]);
	srand(time(NULL));

	out = fopen(filename, "w");

	for(i = 0; i < nr; i++)
	{
		for(j = 0; j <= i; j++)
		{
			if(i == j)
			{
				aux = rand() % upperBoundDIAG;
			}
			else
			{
				aux = rand() % (upperBound + 1 - (-1)*upperBound) - upperBound;
			}
			fprintf(out,"%lf ", aux);
		}
		fprintf(out,"\n");		
	}
	fclose(out);
	printf("\nFile %s generated and closed...\n", filename);
	return 0;	
}
