int infMatInv(double **mat, double **inv, int dim)
{
	int i, j, k;
	double multiplier, **aux;
	if(malloc2ddouble(&aux, dim, dim) != 0)
	{
		return -5;
	}
	//Initialize I matrix
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			aux[i][j] = mat[i][j];
			if(i == j)
			{
				inv[i][j] = 1;
			}
			else
			{
				inv[i][j] = 0;
			}
		}
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j <= i; j++)
		{
			if(i == j)
			{
				multiplier = aux[i][i];
				for(k = 0; k <= i; k++)
				{
					aux[i][k] /= multiplier;
					inv[i][k] /= multiplier;
				}
			}
			else
			{
				multiplier = aux[i][j];
				for(k = 0; k < i; k++)
				{
					aux[i][k] = aux[i][k] - multiplier * aux[j][k];
					inv[i][k] = inv[i][k] - multiplier * inv[j][k];
				}
			}
		}
	}
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if((i == j && fabs(aux[i][j]-1) > EPS) || (i != j && fabs(aux[i][j]) > EPS))
			{
				return -2;
			}
		}
	}
	return 0;
}


int supMatInv(double **mat, double **inv, int dim)
{
	int i, j, k;
	double multiplier, **aux;
	if(malloc2ddouble(&aux, dim, dim) != 0)
	{
		return -5;
	}
	//Initialize I matrix
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			aux[i][j] = mat[i][j];
			if(i == j)
			{
				inv[i][j] = 1;
			}
			else
			{
				inv[i][j] = 0;
			}
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = dim - 1; j >= i; j--)
		{
			if(i == j)
			{
				multiplier = aux[i][i];
				for(k = dim - 1; k >= i; k--)
				{
					aux[i][k] /= multiplier;
					inv[i][k] /= multiplier;
				}
			}
			else
			{
				multiplier = aux[i][j];
				for(k = dim - 1; k > i; k--)
				{
					aux[i][k] = aux[i][k] - multiplier * aux[j][k];
					inv[i][k] = inv[i][k] - multiplier * inv[j][k];
				}
			}
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = dim - 1; j >= 0; j--)
		{
			if((i == j && fabs(aux[i][j]-1) > EPS) || (i != j && fabs(aux[i][j]) > EPS))
			{
				return -2;
			}
		}
	}
	return 0;
}
