/**
 * Author: Hongyan Wang
 * Date: Nov 22, 2016
 */

#ifndef MAT_VEC_H_INCLUDED
#define MAT_VEC_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

typedef struct {
    int nrows, ncols;
    double * d;
} mat;


typedef struct {
    int nrows;
    double * d;
} vec;

/*****************************************************************************
 * Basic matrix and vector operations
 * ***************************************************************************/


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

/* Multiplies matrix M by vector x; returns resulting vector y */
vec * matrix_vec_mult(mat *M, vec *x);

/* return vector dot product */
double vec_dot_product(vec *v1, vec *v2);

/* return a new vector with same elements to existing vector */
vec* copy_vector(vec * v);

/*****************************************************************************
 * GSL matrix and vector operations
 * ***************************************************************************/

/* write matrix to file 
format:
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/

void gsl_matrix_write_to_text_file(gsl_matrix *M, char *fname);

/* load matrix from file 
format:
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/
gsl_matrix * gsl_matrix_load_from_text_file(char *fname);

/* load matrix from binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
gsl_matrix * gsl_matrix_load_from_binary_file(char *fname);

/* write matrix to binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void gsl_matrix_write_to_binary_file(gsl_matrix *M, char *fname);

/* load vector from file 
format:
num_rows
value
.....
value
*/
gsl_vector * gsl_vector_load_from_text_file(char *fname);

/* frobenius norm */
double gsl_get_matrix_frobenius_norm(gsl_matrix *M);


/* C = A*B */
void gsl_matrix_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C);


/* C = A^T*B */
void gsl_matrix_transpose_matrix_mult(gsl_matrix *A, gsl_matrix *B, 
	gsl_matrix *C);


/* y = M*x */
void gsl_matrix_vector_mult(gsl_matrix *M, gsl_vector *x, gsl_vector *y);


/* y = M^T*x */
void gsl_matrix_transpose_vector_mult(gsl_matrix *M, gsl_vector *x, 
	gsl_vector *y);


/* compute compact QR factorization 
M is mxn; Q is mxk and R is kxk
*/
void gsl_compute_QR_compact_factorization(gsl_matrix *M, gsl_matrix *Q, 
	gsl_matrix *R);

#endif