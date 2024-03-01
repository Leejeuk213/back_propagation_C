#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "functions.h"

double ** w2;
double ** b2;
double ** w3;
double ** b3;
double ** w4;
double ** b4;
double ** a1;
double ** z1;
double ** a2;
double ** z2;
double ** a3;
double ** z3;
double ** a4;
double ** z4;


// hyper parameter 설정
int input_nodes = 256; // 16 x 16
int hidden_nodes = 96;
int second_hidden_nodes = 48;

int output_nodes = 7;// t, u, v, x, y, z
double learning_rate = 0.002;
int epochs = 20;


double forward(double **input_data, double **target_data){
    double delta = 1e-7; // log 무한대 발산 방지

    // cross entropy loss 결과 담을 변수들 동적 할당
    double ** result1 = (double**)malloc(1 * sizeof(double*));
    result1[0] = (double*)malloc(output_nodes * sizeof(double));

    double ** result2 = (double**)malloc(1 * sizeof(double*));
    result2[0] = (double*)malloc(output_nodes * sizeof(double));

    double ** result = (double**)malloc(1 * sizeof(double*));
    result[0] = (double*)malloc(output_nodes * sizeof(double));

    // 처음 input data를 z1, a1에 넣어줌
    for(int i = 0; i < input_nodes; i++ ){
        z1[0][i] = input_data[0][i];
        a1[0][i] = input_data[0][i];
    }

    // forward propagation

    // z2 = a1 * w2 + b2
    // a2 = relu(z2) 
    dot_product(z2,a1,w2,1,input_nodes,hidden_nodes);
    elementwise_add(z2,z2,b2,1,hidden_nodes);
    relu(a2,z2,1,hidden_nodes);

    // z3 = a2 * w3 + b3
    // a3 = relu(z3)
    dot_product(z3,a2,w3,1,hidden_nodes,second_hidden_nodes);
    elementwise_add(z3,z3,b3,1,second_hidden_nodes);
    relu(a3,z3,1,second_hidden_nodes);

    // z4 = a3 * w4 + b4
    // a4 = softmax(z4)
    dot_product(z4,a3,w4,1,second_hidden_nodes,output_nodes);
    elementwise_add(z4,z4,b4,1,output_nodes);
    softmax(a4,z4,1,output_nodes);

    // cross entropy loss 
    double ** log_result = (double**)malloc(1 * sizeof(double*));
    log_result[0] = (double*)malloc(output_nodes * sizeof(double));

    // a4 + delta 발산 방지
    add_matrix(a4,1,output_nodes,delta);
    // log(a4) = log_result
    log_matrix(log_result,a4,1,output_nodes);
    // target_data * log(a4) = target_data * log_result = result1
    elementwise_mul(result1,target_data,log_result,1,output_nodes);

    double ** a = (double**)malloc(1 * sizeof(double*));
    a[0] = (double*)malloc(output_nodes * sizeof(double));
    double ** b = (double**)malloc(1 * sizeof(double*));
    b[0] = (double*)malloc(output_nodes * sizeof(double));

    // 1 - target_data a에 넣어줌
    matrix_minus_from_one(a,target_data,1,output_nodes);
    // 1 - a4 b에 넣어줌
    matrix_minus_from_one(b,a4,1,output_nodes);
    // 1 - a4 + delta 발산 방지
    add_matrix(b,1,output_nodes,delta);
    // log(1 - a4)
    log_matrix(b,b,1,output_nodes);
    // (1 - target_data) * log(1 - a4) = a * b = result2
    elementwise_mul(result2,a,b,1,output_nodes);
    // result = target_data * log(a4) + (1 - target_data) * log(1 - a4) = result1 + result2
    elementwise_add(result,result1,result2,1,output_nodes);

    free(result1);
    free(result2);
    free(log_result);
    free(a);
    free(b);

    // loss의 평균
    return -1 * sum(result,1,output_nodes) / output_nodes;
}

void train(double **input_data, double **target_data){

    // forward propagation
    forward(input_data,target_data);

    double ** loss4 = (double**)malloc(1 * sizeof(double*));
    loss4[0] = (double*)malloc(output_nodes * sizeof(double));
    double ** loss3 = (double**)malloc(1 * sizeof(double*));
    loss3[0] = (double*)malloc(second_hidden_nodes * sizeof(double));
    double ** loss2 = (double**)malloc(1 * sizeof(double*));
    loss2[0] = (double*)malloc(hidden_nodes * sizeof(double));

    // back propagation
    //  loss4 = 1/n(a4 - target_data) = dl/da4 * da4/dz4 = dl/dz4
    elementwise_minus(loss4,a4,target_data,1,output_nodes);
    divide_matrix(loss4,1,output_nodes,output_nodes);

    // 미리 연산 결과를 담을 변수들 동적 할당
    double ** a_T = (double**)malloc(second_hidden_nodes * sizeof(double*));
    for(int i = 0; i< second_hidden_nodes; i++) a_T[i] = (double*)malloc(1 * sizeof(double));

    double ** dw = (double**)malloc(second_hidden_nodes * sizeof(double*));
    for(int i = 0; i< second_hidden_nodes; i++) dw[i] = (double*)malloc(output_nodes * sizeof(double));

    double ** db = (double**)malloc(1 * sizeof(double*));
    db[0] = (double*)malloc(output_nodes * sizeof(double));

    // z4 = w4 * a3 + b4 -> dz4/dw4 = a3 
    // 연산을 위해 a3 transpose
    matrix_transpose(a_T,a3,1,second_hidden_nodes);
    // dw4 = dl/dz4 * dz4/dw4 = dl/dw4
    dot_product(dw,a_T,loss4,second_hidden_nodes,1,output_nodes);
    // dw4 = dl/dw4 * learning_rate 
    mul_matrix(dw,dw,second_hidden_nodes,output_nodes,learning_rate);
    // w4 = w4 - dw4
    elementwise_minus(w4,w4,dw,second_hidden_nodes,output_nodes);

    // db4 = dl/dz4 * dz4/db4 = dl/db4, dl/db4 = 1
    mul_matrix(db,loss4,1,output_nodes,learning_rate);
    // b4 = b4 - db4
    elementwise_minus(b4,b4,db,1,output_nodes);

    free(a_T);
    free(dw);
    free(db);

    double ** w_T = (double**)malloc(output_nodes * sizeof(double*));
    for(int i = 0; i< output_nodes; i++) w_T[i] = (double*)malloc(second_hidden_nodes * sizeof(double));

    double ** da = (double**)malloc(1 * sizeof(double*));
    da[0] = (double*)malloc(second_hidden_nodes * sizeof(double));

    double ** dz = (double**)malloc(1 * sizeof(double*));
    dz[0] = (double*)malloc(second_hidden_nodes * sizeof(double));

    // z4 = w4 * a3 + b4 -> dz4/da3 = w4
    // 계산을 위해 w4 transpose
    matrix_transpose(w_T,w4,second_hidden_nodes,output_nodes);
    // dl/dz4 * dz4/da3 = dl/da3
    dot_product(da,loss4,w_T,1,output_nodes,second_hidden_nodes);

    // relu_back = da3/dz3 
    relu_back(dz,a3,1,second_hidden_nodes);

    // loss3 = dl/da3 * da3/dz3 = dl/dz3
    elementwise_mul(loss3,da,dz,1,second_hidden_nodes);


    a_T = (double**)malloc(hidden_nodes * sizeof(double*));
    for(int i = 0; i< hidden_nodes; i++) a_T[i] = (double*)malloc(1* sizeof(double));

    dw = (double**)malloc(hidden_nodes * sizeof(double*));
    for(int i = 0; i< hidden_nodes; i++) dw[i] = (double*)malloc(second_hidden_nodes * sizeof(double));

    db = (double**)malloc(1 * sizeof(double*));
    db[0] = (double*)malloc(second_hidden_nodes * sizeof(double));

    // z3 = w3 * a2 + b3 -> dz3/dw3 = a2
    // 계산을 위해 a2 transpose
    matrix_transpose(a_T,a2,1,hidden_nodes);
    // dw3 = dl/dz3 * dz3/dw3 = dl/dw3
    dot_product(dw,a_T,loss3,hidden_nodes,1,second_hidden_nodes);
    // dw3 = dl/dw3 * learning_rate
    mul_matrix(dw,dw,hidden_nodes,second_hidden_nodes,learning_rate);
    // w3 = w3 - dw3
    elementwise_minus(w3,w3, dw,hidden_nodes,second_hidden_nodes);

    // db3 = dl/dz3 * dz3/db3 = dl/db3, dl/db3 = 1
    mul_matrix(db,loss3,1,second_hidden_nodes,learning_rate);
    // b3 = b3 - db3
    elementwise_minus(b3,b3,db,1,second_hidden_nodes);

    free(w_T);
    free(da);
    free(dz);
    free(a_T);
    free(dw);
    free(db);

    w_T = (double**)malloc(second_hidden_nodes * sizeof(double*));
    for(int i = 0; i< second_hidden_nodes; i++) w_T[i] = (double*)malloc(hidden_nodes * sizeof(double));

    da = (double**)malloc(1 * sizeof(double*));
    da[0] = (double*)malloc(hidden_nodes * sizeof(double));

    dz = (double**)malloc(1 * sizeof(double*));
    dz[0] = (double*)malloc(hidden_nodes * sizeof(double));

    // z3 = w3 * a2 + b3 -> dz3/da2 = w3
    // 계산을 위해 w3 transpose
    matrix_transpose(w_T,w3,hidden_nodes,second_hidden_nodes);
    // dl/dz3 * dz3/da2 = dl/da2
    dot_product(da,loss3,w_T,1,second_hidden_nodes,hidden_nodes);
    // relu_back = da2/dz2
    relu_back(dz,a2,1,hidden_nodes);
    // loss2 = dl/da2 * da2/dz2 = dl/dz2
    elementwise_mul(loss2,da,dz,1,hidden_nodes);

    a_T = (double**)malloc(input_nodes * sizeof(double*));
    for(int i = 0; i< input_nodes; i++) a_T[i] = (double*)malloc(1* sizeof(double));

    dw = (double**)malloc(input_nodes * sizeof(double*));
    for(int i = 0; i< input_nodes; i++) dw[i] = (double*)malloc(hidden_nodes * sizeof(double));

    db = (double**)malloc(1 * sizeof(double*));
    db[0] = (double*)malloc(hidden_nodes * sizeof(double));

    // z2 = w2 * a1 + b2 -> dz2/dw2 = a1
    // 계산을 위해 a1 transpose
    matrix_transpose(a_T,a1,1,input_nodes);
    // dw2 = dl/dz2 * dz2/dw2 = dl/dw2
    dot_product(dw,a_T,loss2,input_nodes,1,hidden_nodes);
    // dw2 = dl/dw2 * learning_rate
    mul_matrix(dw,dw,input_nodes,hidden_nodes,learning_rate);
    // w2 = w2 - dw2
    elementwise_minus(w2,w2,dw,input_nodes,hidden_nodes);

    // db2 = dl/dz2 * dz2/db2 = dl/db2, dl/db2 = 1
    mul_matrix(db,loss2,1,hidden_nodes,learning_rate);
    // b2 = b2 - db2
    elementwise_minus(b2,b2,db,1,hidden_nodes);

    free(w_T);
    free(da);
    free(dz);
    free(a_T);
    free(dw);
    free(db);

    free(loss2);
    free(loss3);
    free(loss4);
}

int predict(double ** input_data){

    double ** Z2;
    double ** A2;
    double ** Z3;
    double ** A3;
    double ** Z4;
    double ** A4;

    Z2 = (double**)malloc(1 * sizeof(double*)); 
    Z2[0] = (double*)malloc(hidden_nodes * sizeof(double));
    zeros(Z2,1,hidden_nodes);
    
    A2 = (double**)malloc(1 * sizeof(double*)); 
    A2[0] = (double*)malloc(hidden_nodes * sizeof(double));
    zeros(A2,1,hidden_nodes);

    Z3 = (double**)malloc(1 * sizeof(double*)); 
    Z3[0] = (double*)malloc(second_hidden_nodes * sizeof(double));
    zeros(Z3,1,second_hidden_nodes);
    
    A3 = (double**)malloc(1 * sizeof(double*)); 
    A3[0] = (double*)malloc(second_hidden_nodes * sizeof(double));
    zeros(A3,1,second_hidden_nodes);

    Z4 = (double**)malloc(1 * sizeof(double*)); 
    Z4[0] = (double*)malloc(output_nodes * sizeof(double));
    zeros(Z4,1,output_nodes);
    
    A4 = (double**)malloc(1 * sizeof(double*)); 
    A4[0] = (double*)malloc(output_nodes * sizeof(double));
    zeros(A4,1,output_nodes);

    // 기본적인 구조 forward와 동일 
    dot_product(Z2,input_data,w2,1,input_nodes,hidden_nodes);
    elementwise_add(Z2,Z2,b2,1,hidden_nodes);
    relu(A2,Z2, 1, hidden_nodes);

    dot_product(Z3,A2,w3,1,hidden_nodes,second_hidden_nodes);
    elementwise_add(Z3,Z3,b3,1,second_hidden_nodes);
    relu(A3,Z3, 1, second_hidden_nodes);
    
    dot_product(Z4,A3,w4,1,second_hidden_nodes,output_nodes);
    elementwise_add(Z4,Z4,b4,1,output_nodes);
    softmax(A4,Z4, 1, output_nodes);

    // 출력값 중 가장 큰 값의 인덱스를 리턴
    int predicted_num = find_max_index(A4,output_nodes);

    free(Z2);
    free(A2);
    free(Z3);
    free(A3);
    free(Z4);
    free(A4);

    return predicted_num;
}

void accuracy(double input_data[140][257]){

    double count = 0;
    double ** data= (double**)malloc(1 * sizeof(double*));
    data[0] = (double*)malloc(input_nodes * sizeof(double));

    // test data 140개를 predict 함수를 통해 예측한 값과 label이 같으면 count를 증가시킴
    for(int i = 0; i< 140; i++){
        int label = input_data[i][256];
        for(int j= 0; j<256; j++){
            data[0][j] = (input_data[i][j]/255.0 *0.99) + 0.01;
        }
        int predicted_num = predict(data);

        if (label == predicted_num) count ++;
    }
    printf("count %lf, Current Accuracy = %lf\n",count, 100*(count/140));
    free(data);
}

void init()
{
    // input layer, hidden layer, output layer 선형회귀 값, 출력 값 선언 및 가중치 설정  python에서의 class init부분

    w2 = (double**)malloc(input_nodes * sizeof(double*)); 
    for (int i = 0; i < input_nodes; i++) w2[i] = (double*)malloc(hidden_nodes * sizeof(double));
    randnArray(w2,input_nodes,hidden_nodes);
    divide_matrix(w2,input_nodes,hidden_nodes,sqrt(input_nodes/2)); // he 초기화

    b2 = (double**)malloc(1 * sizeof(double*)); 
    b2[0] = (double*)malloc(hidden_nodes * sizeof(double));
    randArray(b2,1,hidden_nodes);

    w3 = (double**)malloc(hidden_nodes * sizeof(double*)); 
    for (int i = 0; i < hidden_nodes; i++) w3[i] = (double*)malloc(second_hidden_nodes * sizeof(double));
    randnArray(w3,hidden_nodes,second_hidden_nodes);
    divide_matrix(w3,hidden_nodes,second_hidden_nodes,sqrt(hidden_nodes/2));

    b3 = (double**)malloc(1 * sizeof(double*)); 
    b3[0] = (double*)malloc(second_hidden_nodes * sizeof(double));
    randArray(b3,1,second_hidden_nodes);

    w4 = (double**)malloc(second_hidden_nodes * sizeof(double*)); 
    for (int i = 0; i < second_hidden_nodes; i++) w4[i] = (double*)malloc(output_nodes* sizeof(double));
    randnArray(w4,second_hidden_nodes,output_nodes);
    divide_matrix(w4,second_hidden_nodes,output_nodes,sqrt(second_hidden_nodes/2));

    b4 = (double**)malloc(1 * sizeof(double*)); 
    b4[0] = (double*)malloc(output_nodes * sizeof(double));
    randArray(b4,1,output_nodes);

    // z, a 동적 할당 및 초기화
    z1 = (double**)malloc(1 * sizeof(double*)); 
    z1[0] = (double*)malloc(input_nodes * sizeof(double));
    zeros(z1,1,input_nodes);
    
    a1 = (double**)malloc(1 * sizeof(double*)); 
    a1[0] = (double*)malloc(input_nodes * sizeof(double));
    zeros(a1,1,input_nodes);

    z2 = (double**)malloc(1 * sizeof(double*)); 
    z2[0] = (double*)malloc(hidden_nodes * sizeof(double));
    zeros(z2,1,hidden_nodes);
    
    a2 = (double**)malloc(1 * sizeof(double*)); 
    a2[0] = (double*)malloc(hidden_nodes * sizeof(double));
    zeros(a2,1,hidden_nodes);

    z3 = (double**)malloc(1 * sizeof(double*)); 
    z3[0] = (double*)malloc(second_hidden_nodes * sizeof(double));
    zeros(z3,1,second_hidden_nodes);
    
    a3 = (double**)malloc(1 * sizeof(double*)); 
    a3[0] = (double*)malloc(second_hidden_nodes * sizeof(double));
    zeros(a3,1,second_hidden_nodes);

    z4 = (double**)malloc(1 * sizeof(double*)); 
    z4[0] = (double*)malloc(output_nodes * sizeof(double));
    zeros(z4,1,output_nodes);
    
    a4 = (double**)malloc(1 * sizeof(double*)); 
    a4[0] = (double*)malloc(output_nodes * sizeof(double));
    zeros(a4,1,output_nodes);
}


int main() {

    // 시드설정 가중치를 위한
    srand(time(NULL));  

    // test파일 train 파일 불러오기
    FILE *file;
    char buffer[1024]; 
    int i, j;
    double test_data[140][257]; 
    double train_data[315][257]; 

    file = fopen("test.csv", "r");
    i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        char *token = strtok(buffer, ",");
        j = 0;
        while (token != NULL) {
            test_data[i][j] = atoi(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(file);

    file = fopen("train.csv", "r");
    i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        char *token = strtok(buffer, ",");
        j = 0;
        while (token != NULL) {
            train_data[i][j] = atoi(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    // 파일 닫기
    fclose(file);
        
    init();

    for(int i = 0; i < epochs; i++){

        for(int step = 0; step < 315; step++){

            // target data, input data 동적 할당 및 초기화
            double ** target_data = (double**)malloc(1 * sizeof(double*));
            target_data[0] = (double*)malloc(output_nodes * sizeof(double));
            zeros(target_data,1,output_nodes);
            // 0.01로 초기화
            add_matrix(target_data,1,output_nodes,0.01);
            
            double ** input_data = (double**)malloc(1 * sizeof(double*));
            input_data[0] = (double*)malloc(input_nodes * sizeof(double));
            int label = train_data[step][256];
            // one hot encoding 
            // 정답 레이블에 해당하는 인덱스에 0.99를 넣어줌 나머지는 0.01
            target_data[0][label] = 0.99;

            for(int j = 0; j < 256; j++){
                // 이미지 데이터 0~255 -> 0.01~1.0 
                // 정규화
                input_data[0][j] = (train_data[step][j]/255.0 *0.99) + 0.01;
            }
            train(input_data,target_data);

            // 계산 도중에 z2의 값이 무한대로 갔음 0.00001으로 나눈 것을 0으로 인식해버림 가중치 재설정하고 학습 처음부터 다시 시작함.
            // 초기 한번만 CHECK하면 이후에 무한대로 가는 경우는 없었음. 따라서 이런 식으로 하드코딩
            for(int i=0; i< hidden_nodes; i++){
                if( isinf(z2[0][i]) ){
                    free(w2);
                    free(b2);
                    free(w3);
                    free(b3);
                    free(w4);
                    free(b4);
                    free(a1);
                    free(z1);
                    free(a2);
                    free(z2);
                    free(a3);
                    free(z3);
                    free(a4);
                    free(z4);
                    init();
                    i = 0;
                    step = -1;
                }
            }
            // 10번마다 loss 출력
            if(step % 10 == 0){
                printf("current epochs = %d step = %d, loss_val = %lf\n",i,step,forward(input_data,target_data));
            }

            free(target_data);
            free(input_data);
        }
        // 한번 학습이 끝날 때마다 정확도 출력
        accuracy(test_data);
    }

    free(w2);
    free(b2);
    free(w3);
    free(b3);
    free(w4);
    free(b4);
    free(a1);
    free(z1);
    free(a2);
    free(z2);
    free(a3);
    free(z3);
    free(a4);
    free(z4);

    return 0;
}