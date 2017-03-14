#ifndef MATMPIOP_H
#define MATMPIOP_H

void (*printVector)(int *, int );
void (*printMatrix)(int **, int , int );

int (*malloc2dint)(int ***, int , int );
int (*free2dint)(int ***);

int** (*sumM)(int **, int **, int , int , int );
int** (*prodM)(int **, int , int , int **, int , int , int );

#endif
