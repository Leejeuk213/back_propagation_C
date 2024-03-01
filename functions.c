#include<stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES   
#include <math.h>

#include "functions.h"

// 모든 요소를 0으로 초기화
void zeros(double ** array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = 0;
        }
    }
}

// 정규 분포 난수 생성 함수
double randn() {
    double u1 = (double)rand() / RAND_MAX; // 0에서 1 사이의 난수 생성
    double u2 = (double)rand() / RAND_MAX; // 0에서 1 사이의 또 다른 난수 생성
    double z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2); // 정규 분포로 변환

    return z;
}

// 정규 분포 난수 행렬 생성 함수
void randnArray(double ** array,int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = randn(); // 정규 분포 난수로 초기화
        }
    }
}

// 0에서 1 사이의 난수 행렬 생성 함수
void randArray(double ** array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// 행렬을 주어진 값으로 나누는 함수
void divide_matrix(double** array, int rows, int cols, double value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = array[i][j] / value;
        }
    }
}

// 행렬을 주어진 값으로 곱하는 함수
void mul_matrix(double** array1,double** array2, int rows, int cols, double value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array1[i][j] = array2[i][j] * value;
        }
    }
}

// 행렬을 주어진 값으로 더하는 함수
void add_matrix(double** array, int rows, int cols, double value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = array[i][j] + value;
        }
    }
}

// a x b * b x c 행렬임 각각 a, b, c 추출  => a x c 사이즈의 행렬 리턴
void dot_product(double ** array, double ** array1, double **array2, int a, int b, int c ){
    
    for(int i = 0; i < a; i++) {
        for (int j = 0; j < c; j++){
            double product = 0;
            for (int k = 0; k < b; k++){
                product += array1[i][k] * array2[k][j] ; 
            }
            array[i][j] = product;
        }
    }
}

// sigmoid 함수
void sigmoid(double ** array1, double ** array2, int row, int col ){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            array1[i][j] = 1 / (1 + exp(-1 * array2[i][j]) );
        }
    }
}

// relu 함수
void relu(double ** array1, double ** array2, int row, int col ){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(array2[i][j] > 0)  array1[i][j] = array2[i][j];
            else array1[i][j] = 0; 
        }
    } 
}

// softmax 함수
void softmax(double ** array1, double ** array2, int row, int col ){

    double max_value = array2[0][find_max_index(array2, col)];
    double sum = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            sum += exp(array2[i][j]-max_value);
        }
    } 
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            array1[i][j] = exp(array2[i][j]-max_value) / sum;
        }
    } 
}

// 행렬의 각 요소에 log 취하는 함수
void log_matrix(double ** array1,double ** array2, int row, int col ){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            array1[i][j] = log(array2[i][j]);
        }
    }
}

// 행렬의 요소끼리 곱하는 함수
void elementwise_mul(double ** array, double ** array1, double **array2, int row, int col){
    for(int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++){
            array[i][j] = array1[i][j] * array2[i][j];
        }
    }
}

// 행렬의 요소끼리 더하는 함수
void elementwise_add(double ** array, double ** array1, double **array2, int row, int col){
    
    for(int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++){
            array[i][j] = array1[i][j] + array2[i][j];
        }
    }
}

// 행렬의 요소끼리 빼는 함수
void elementwise_minus(double ** array, double ** array1, double **array2, int row, int col){
    
    for(int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++){
            array[i][j] = array1[i][j] - array2[i][j];
        }
    }
}

// 행렬의 요소를 1에서 빼는 함수
void matrix_minus_from_one(double ** array1,double ** array2, int row, int col ){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            array1[i][j] = 1 - array2[i][j];
        }
    }
}

// 행렬의 모든 요소의 합을 구하는 함수
double sum(double ** array,int row, int col){
    
    double result = 0;
    for(int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++){
            result += array[i][j];
        }
    }
    free(array);
    return result;
}

// 행렬 전치 함수 (행과 열을 바꿈)
void matrix_transpose(double ** result, double** matrix, int row, int col) {
    // 전치된 행렬의 행과 열 수는 원본 행렬의 열과 행 수와 같음

    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            result[i][j] = matrix[j][i];
        }
    }
}

// 행렬의 열에서 가장 큰 값의 인덱스를 구하는 함수
int find_max_index(double ** array, int col){
    int max_index = 0 ;
    double max_value = array[0][0];

        for(int i = 1; i < col; i++){
            if(array[0][i] > max_value){
                max_index = i;
                max_value = array[0][i];
            }
        }
    return max_index;
}

// relu 함수의 역함수
void relu_back(double ** result_array, double ** array, int row, int col ){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(array[i][j] > 0)  result_array[i][j] = 1;
            else result_array[i][j] = 0; 
        }
    }
}

// 배열 출력 함수 (테스트용)
void printArray(double** array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}