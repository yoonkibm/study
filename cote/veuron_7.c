#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double input_a[][3] = {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}};

double* proj(int size, double* u, double* a){

    double numerator = 0;
    double denominator = 0;

    double* result = malloc(size*sizeof(double));

    for(int i = 0; i < size; i++){
        denominator += u[i]*u[i];
        numerator += u[i]*a[i];
    }

    for(int i = 0; i < size; i++){
        result[i] = (numerator/denominator)*u[i];
    }

    return result;
}

int main(){
    size_t rows = sizeof(input_a) / sizeof(input_a[0]);
    size_t cols = sizeof(input_a[0]) / sizeof(input_a[0][0]);

    double** U = malloc(cols * sizeof(double*));
    double** A = malloc(cols * sizeof(double*));
    double** E = malloc(cols * sizeof(double*));
    double** Q = malloc(rows * sizeof(double*));
    double** R = malloc(cols *sizeof(double*));
    for(int i = 0; i < cols; i++){
        U[i] = malloc(rows * sizeof(double));
        A[i] = malloc(rows * sizeof(double));
        E[i] = malloc(rows * sizeof(double));
        R[i] = malloc(cols * sizeof(double));
    }

    for(int i = 0; i < rows; i++){
        Q[i] = malloc(cols * sizeof(double));
    }

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            A[j][i] = input_a[i][j];
        }
    }

    for(int k = 0; k < cols; k++){
        for (int i = 0; i < rows; i++){
            U[k][i] = A[k][i];
        }

        for(int j = 0; j < k; j++){
            double* p = proj(rows, U[j], A[k]);
            for(int i = 0; i < rows; i++){
                U[k][i] -= p[i];
            }
            free(p);
        }
    }

    for(int i = 0; i < cols; i++){
        double norm2_ui = 0;

        for(int j = 0; j < rows; j++){
            norm2_ui += U[i][j]*U[i][j];
        }

        for(int j = 0; j < rows; j++){
            E[i][j] = U[i][j]/sqrt(norm2_ui);
        }
    }

    for(int i = 0; i < cols; i++){
        for(int j = 0; j < rows; j++){
            Q[j][i] = E[i][j];
        }
    }

    for(int i = 0; i < cols; i++){
        for(int j = 0; j < cols; j++){
            double r = 0;
            if(i <= j){
                for(int k = 0; k < rows; k++){
                    r += Q[k][i]*A[j][k];
                }
                R[i][j] = r;
            }
            else R[i][j] = r;
        }
    }

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%f ", input_a[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%f ", Q[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    for(int i = 0; i < cols; i++){
        for(int j = 0; j < cols; j++){
            printf("%f ", R[i][j]);
        }
        printf("\n");
    }

    for(int i = 0; i < cols; ++i){
        free(U[i]);
        free(A[i]);
        free(E[i]);
        free(R[i]);
    }

    for(int i = 0; i < rows; i++){
        free(Q[i]);
    }

    free(U);
    free(A);
    free(E);
    free(R);
    free(Q);

    return 0;
}