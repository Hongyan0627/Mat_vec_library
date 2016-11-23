/**
 * Author: Hongyan Wang
 * Date: Nov 22, 2016
 */

#ifndef MAT_VEC_H_INCLUDED
#define MAT_VEC_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int nrows, ncols;
    double * d;
} mat;


typedef struct {
    int nrows;
    double * d;
} vec;

/* initialize new matrix and set all entries to zero */
mat * matrix_new(int nrows, int ncols);

/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows);

/* delete matrix and clean memory*/
void matrix_delete(mat *M);

/* delete vector and clean memory*/
void vector_delete(vec *v);

/* get element from a matrix, column major format*/
double matrix_get_element(mat *M, int row_num, int col_num);

/* get element from a vector */
double vector_get_element(vec *v, int row_num);

/* set element for a matrix, column major format */
void matrix_set_element(mat *M, int row_num, int col_num, double val);

/* set element for a vector */
void vector_set_element(vec *v, int row_num, double val);

/* print a matrix */
void matrix_print(mat * M);

/* print a vector */
void vector_print(vec * v);

/* load matrix from binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
mat * matrix_load_from_binary_file(char *fname);

/* column vector load from binary file 
 * the nonzeros are in order of a loop over all rows  
format:
num_rows (int) 
nnz (double)
...
nnz (double)
*/
vec * vector_load_from_binary_file(char *fname);

/* write matrix to binary file 
 * the nonzeros are in order of double loop over all rows of each column 
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void matrix_write_to_binary_file(mat *M, char *fname);

/* column vector write to binary file 
 * the nonzeros are in order of a loop over all rows  
format:
num_rows (int) 
nnz (double)
...
nnz (double)
*/
void vector_write_to_binary_file(vec *v, char *fname);

/* get vector euclidean norm*/
double get_vector_euclidean_norm(vec *v);


#endif