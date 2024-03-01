#ifndef FUNCTIONS_H
#define FUNCTIONS_H

void zeros(double ** array,int rows, int cols) ;
double randn();
void randnArray(double ** array, int rows, int cols) ;
void randArray(double ** array, int rows, int cols);
void divide_matrix(double** array, int rows, int cols, double value);
void mul_matrix(double** array1,double** array2, int rows, int cols, double value) ;
void add_matrix(double** array, int rows, int cols, double value);
void dot_product(double ** array, double ** array1, double **array2, int a, int b, int c ); 
void sigmoid(double ** array1, double ** array2, int row, int col );
void relu(double ** array1, double ** array2, int row, int col );
void softmax(double ** array1, double ** array2, int row, int col );
void log_matrix(double ** array1,double ** array2, int row, int col ) ;
void elementwise_mul(double ** array, double ** array1, double **array2, int row, int col) ;
void elementwise_minus(double ** array, double ** array1, double **array2, int row, int col);
void elementwise_add(double ** array, double ** array1, double **array2, int row, int col) ;
void matrix_minus_from_one(double ** array1,double ** array2, int row, int col ) ;
double sum(double ** array,int row, int col);
void matrix_transpose(double ** result, double** matrix, int row, int col) ;
void relu_back(double ** result_array, double ** array, int row, int col ) ;
int find_max_index(double ** array, int col);
void printArray(double** array, int rows, int cols);

#endif