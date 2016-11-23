#include "mat_vec_gsl.h"

int main()
{
  int m=5,n=4;
  int i, j;
  double val;
  gsl_spmatrix *A = gsl_spmatrix_alloc(m, n); /* triplet format */
  gsl_spmatrix *B, *C;
  gsl_vector *x,*y,*v;

  /* build the sparse matrix */
  gsl_spmatrix_set(A, 0, 2, 3.1);
  gsl_spmatrix_set(A, 0, 3, 4.6);
  gsl_spmatrix_set(A, 1, 0, 1.0);
  gsl_spmatrix_set(A, 1, 2, 7.2);
  gsl_spmatrix_set(A, 3, 0, 2.1);
  gsl_spmatrix_set(A, 3, 1, 2.9);
  gsl_spmatrix_set(A, 3, 3, 8.5);
  gsl_spmatrix_set(A, 4, 0, 4.1);

  //   /* build the vector */
  //   v = gsl_vector_calloc(n);   
  //   for(i=0; i<n; i++){
  //       gsl_vector_set(v,i,1.0);
  //   }

  //   val = gsl_vector_get(v,2);
  //   printf("val = %f\n", val);  


  //   x = gsl_vector_calloc(m);   
  //   for(i=0; i<m; i++){
  //       gsl_vector_set(x,i,1.0);
  //   }

  // /* print out in regular format */
  //   printf("matrix A:\n");
  //   gsl_matrix_print(A);
  //   printf("fro norm of A = %f\n", gsl_get_sparse_matrix_frobenius_norm(A));
    
  //   printf("vector v:\n");
  //   gsl_vector_print(v);

  // /* convert to compressed row format */
  // C = gsl_spmatrix_crs(A);

  // printf("matrix in compressed row format:\n");
  // printf("i = [ ");
  // for (i = 0; i < C->nz; ++i)
  //   printf("%zu, ", C->i[i]);
  // printf("]\n");

  // printf("p = [ ");
  // for (i = 0; i < C->size1 + 1; ++i)
  //   printf("%zu, ", C->p[i]);
  // printf("]\n");

  // printf("d = [ ");
  // for (i = 0; i < C->nz; ++i)
  //   printf("%g, ", C->data[i]);
  // printf("]\n");

  //   printf("matrix-vector mult..\n");
  //   gsl_sparse_matrix_vector_mult(C,v,&y);
  //   gsl_vector_print(y);
  //   gsl_vector_free(y);

  //   printf("matrix-transpose vector mult..\n");
  //   gsl_sparse_matrix_transpose_vector_mult(C,x,&y);
  //   gsl_vector_print(y);
  //   gsl_vector_free(y);


  // gsl_spmatrix_free(A);
  // gsl_spmatrix_free(C);

  return 0;
}


