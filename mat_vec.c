/**
 * Author: Hongyan Wang
 * Date: Nov 22, 2016
 */

#include "mat_vec.h"


/*****************************************************************************
 * Basic matrix and vector operations
 * ***************************************************************************/

 /* initialize new matrix and set all entries to zero */
mat * matrix_new(int nrows, int ncols) {
    mat *M = malloc(sizeof(mat));
    M->d = (double*)calloc(nrows*ncols, sizeof(double));
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}

/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows) {
    vec *v = malloc(sizeof(vec));
    v->d = (double*)calloc(nrows,sizeof(double));
    v->nrows = nrows;
    return v;
}

/* delete matrix and clean memory*/
void matrix_delete(mat *M) {
    free(M->d);
    free(M);
}

/* delete vector and clean memory*/
void vector_delete(vec *v) {
    free(v->d);
    free(v);
}

/* get element from a matrix, column major format */
double matrix_get_element(mat *M, int row_num, int col_num) {
    return M->d[col_num*(M->nrows) + row_num];
}

/* get element from a vector */
double vector_get_element(vec *v, int row_num) {
    return v->d[row_num];
}

/* set element for a matrix */
void matrix_set_element(mat *M, int row_num, int col_num, double val) {
    M->d[col_num*(M->nrows) + row_num] = val;
}

/* set element for a vector */
void vector_set_element(vec *v, int row_num, double val) {
    v->d[row_num] = val;
}

/* print a matrix */
void matrix_print(mat * M) {
    int i,j;
    double val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            printf("%f  ", val);
        }
        printf("\n");
    }
}

/* print a vector */
void vector_print(vec * v) {
    int i;
    double val;
    for(i=0; i<v->nrows; i++){
        val = vector_get_element(v, i);
        printf("%f\n", val);
    }
}

/* load matrix from binary file, column major format
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
mat * matrix_load_from_binary_file(char *fname) {
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    mat *M;

    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read m
    fread(&num_columns,sizeof(int),one,fp); //read n
    printf("initializing M of size %d by %d\n", num_rows, num_columns);
    M = matrix_new(num_rows,num_columns);
    printf("done..\n");

    // read and set elements
    for(j=0; j<num_columns; j++){
        for(i=0; i<num_rows; i++){
            fread(&nnz_val,sizeof(double),one,fp); //read nnz
            matrix_set_element(M,i,j,nnz_val);
        }
    }
    fclose(fp);

    return M;
}

/* column vector load from binary file 
 * the nonzeros are in order of a loop over all rows  
format:
num_rows (int) 
nnz (double)
...
nnz (double)
*/
vec * vector_load_from_binary_file(char *fname) {
    int i, j, num_rows, row_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    vec *v;

    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read num_rows
    printf("initializing v of length %d\n", num_rows);
    v = vector_new(num_rows);
    printf("done..\n");

    // read and set elements
    for(i=0; i<num_rows; i++){
        fread(&nnz_val,sizeof(double),one,fp); //read nnz
        vector_set_element(v,i,nnz_val);
    }
    fclose(fp);

    return v;
}

/* write matrix to binary file, column major format
 * the nonzeros are in order of double loop over all rows of each column 
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void matrix_write_to_binary_file(mat *M, char *fname) {
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    num_rows = M->nrows; num_columns = M->ncols;
    
    fp = fopen(fname,"w");
    fwrite(&num_rows,sizeof(int),one,fp); //write m
    fwrite(&num_columns,sizeof(int),one,fp); //write n

    // write the elements
    for(j=0; j<num_columns; j++){
        for(i=0; i<num_rows; i++){
            nnz_val = matrix_get_element(M,i,j);
            fwrite(&nnz_val,sizeof(double),one,fp); //write nnz
        }
    }
    fclose(fp);
}

/* column vector write to binary file 
 * the nonzeros are in order of a loop over all rows  
format:
num_rows (int) 
nnz (double)
...
nnz (double)
*/
void vector_write_to_binary_file(vec *v, char *fname) {
    int i, num_rows, row_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    num_rows = v->nrows;
    
    fp = fopen(fname,"w");
    fwrite(&num_rows,sizeof(int),one,fp); //write m

    // write the elements
    for(i=0; i<num_rows; i++){
        nnz_val = vector_get_element(v,i);
        fwrite(&nnz_val,sizeof(double),one,fp); //write nnz
    }
    fclose(fp);
}

/* get vector euclidean norm*/
double get_vector_euclidean_norm(vec *v) {
    int i, nrows = v->nrows;
    double val,nval = 0;
    for(i=0; i<nrows; i++){
        val = vector_get_element(v,i);
        nval += val*val;
    }   
    return sqrt(nval);
}

/* Multiplies matrix M by vector x; returns resulting vector y */
vec * matrix_vec_mult(mat *M, vec *x)
{
    int i,j;
    double val;
    vec *y;
    y = vector_new(M->nrows);
    for (i = 0; i < M->nrows; i++){
        for (j = 0; j < M->ncols; j++){
            val = matrix_get_element(M,i,j);
            vector_set_element(y,i,vector_get_element(y,i) + 
                val * vector_get_element(x,j));
        }
    }
    return y;
}

/* return vector dot product */
double vec_dot_product(vec *v1, vec *v2){
    double val = 0.0;
    int i;
    for(i = 0; i < v1->nrows; i++){
        val += (vector_get_element(v1,i) * vector_get_element(v2,i));
    }
    return val;
}

/* return a new vector with same elements to existing vector */
vec* copy_vector(vec * v){
    int i;
    vec *y;
    double val;
    y = vector_new(v->nrows);
    for(i = 0; i < v->nrows; i++){
        val = vector_get_element(v, i);
        vector_set_element(y,i,val);
    }
    return y;
}




