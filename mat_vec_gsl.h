/**
 * Author: Hongyan Wang
 * Date: Nov 23, 2016
 */

#ifndef MAT_VEC_GSL_H_INCLUDED
#define MAT_VEC_GSL_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

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
gsl_matrix *gsl_matrix_load_from_text_file(char *fname);

/* load matrix from binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
gsl_matrix *gsl_matrix_load_from_binary_file(char *fname);

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
gsl_vector *gsl_vector_load_from_text_file(char *fname);

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

/* write sparse matrix to binary file assuming row compressed format */ 
void gsl_sparse_matrix_write_to_binary_file(gsl_spmatrix *A, char *fname);

/* write sparse matrix to binary file assuming row compressed format */
gsl_spmatrix *gsl_sparse_matrix_load_from_binary_file(char *fname);

/* load vector from file 
format:
num_rows
value
.....
value
*/
gsl_vector *gsl_vector_load_from_text_file(char *fname);

/* print a sparse matrix out in dense format */
void gsl_sparse_matrix_print(gsl_spmatrix *A);

/* print a column vector */
void gsl_vector_print(gsl_vector *v);

/* frobenius norm of a sparse matrix */
double gsl_get_sparse_matrix_frobenius_norm(gsl_spmatrix *A);

/*  y \leftarrow \alpha (A) v + \beta y */
void gsl_sparse_matrix_vector_mult(gsl_spmatrix *A,gsl_vector *v, 
	gsl_vector **y);

/*  y \leftarrow \alpha transpose(A) v + \beta y */
void gsl_sparse_matrix_transpose_vector_mult(gsl_spmatrix *A,gsl_vector *v, 
	gsl_vector **y);


#endif