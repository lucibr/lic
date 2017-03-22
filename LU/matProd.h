double** prodM_DI(double **mat1, int nrL1, int nrC1, int **mat2, int nrL2, int nrC2)
{
	int i, j, k;
	double **resultMat;
	if(nrC1 != nrL2)
	{
		printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}

	if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
	{
		for(i = 0; i < nrL1; i++)
			for(j = 0; j < nrC2; j++) 
			{
				resultMat[i][j] = 0;
				for(k = 0; k < nrC1; k++)
					resultMat[i][j] += mat1[i][k]*((double)mat2[k][j]);
			}
		return resultMat;		
	}
	else
	{
		//Memory could not be allocated
		return NULL;
	}
	return NULL;
}

double** prodM_ID(int **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2)
{
	double **resultMat;
	int i, j, k;
	if(nrC1 != nrL2)
	{
		printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}

	if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
	{
		for(i = 0; i < nrL1; i++)
			for(j = 0; j < nrC2; j++) 
			{
				resultMat[i][j] = 0;
				for(k = 0; k < nrC1; k++)
				{
					resultMat[i][j] += ((double)mat1[i][k])*mat2[k][j];
				}
			}
		return resultMat;		
	}
	else
	{
		//Memory could not be allocated
		return NULL;
	}
	return NULL;
}

double** prodM_DD(double **mat1, int nrL1, int nrC1, double **mat2, int nrL2, int nrC2)
{
	double **resultMat;
	int i, j, k;
	if(nrC1 != nrL2)
	{
		printf("\nThe matrices cannot be multiplied! (nrC(A) != nrL(B))\n");
		return NULL;
	}
	
	if(malloc2ddouble(&resultMat, nrL1, nrC2) == 0)
	{
		for(i = 0; i < nrL1; i++)
		{
			for(j = 0; j < nrC2; j++) 
			{
				for(k = 0; k < nrC1; k++)
				{
					resultMat[i][j] += mat1[i][k]*mat2[k][j];
				}
			}
		}
		return resultMat;		
	}
	else
	{
		//Memory could not be allocated
		return NULL;
	}
	return NULL;
}
