
void difM_DD(double **mat1, double **mat2, int nrR, int nrC)
{
	int i, j;
	for(i = 0; i < nrR; i++)
	{
		for (j = 0; j < nrC; j++)
		{
			mat1[i][j] -= mat2[i][j];
		}
	}
}
