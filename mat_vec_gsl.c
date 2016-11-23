#include "mat_vec_gsl.h"

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

void gsl_matrix_write_to_text_file(gsl_matrix *M, char *fname) {
    int i, j;
    FILE *fp;
    
    fp = fopen(fname,"w");
    fprintf(fp, "%d %d %d\n", M->size1, M->size2, (M->size1)*(M->size2));
    for(i=0; i<M->size1; i++){
        for(j=0; j<M->size2; j++){
            fprintf(fp, "%d %d %f\n", i, j, gsl_matrix_get(M,i,j));
        }
    }
    fclose(fp);
}

/* load matrix from file 
format:
num_rows num_columns num_nonzeros
row col nnz
.....
row col nnz
*/
gsl_matrix * gsl_matrix_load_from_text_file(char *fname) {
    int i, j, num_rows, num_columns, num_nonzeros, row_num, col_num;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    gsl_matrix *M;
    
    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read dimensions and nnzs 
    sscanf(line, "%d %d %d", &num_rows, &num_columns, &num_nonzeros);

    M = gsl_matrix_calloc(num_rows, num_columns); // calloc sets all elements to zero

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<num_nonzeros; i++){
        fgets(line,100,fp); 
        sscanf(line, "%d %d %s", &row_num, &col_num, nnz_val_str);
        nnz_val = atof(nnz_val_str);
        gsl_matrix_set(M, row_num, col_num, nnz_val);
    }
    fclose(fp);

    // clean
    free(line);
    free(nnz_val_str);

    return M;
}

/* load matrix from binary file, row major format
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
gsl_matrix * gsl_matrix_load_from_binary_file(char *fname) {
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    gsl_matrix *M;
    
    fp = fopen(fname,"r");
    fread(&num_rows,sizeof(int),one,fp); //read m
    fread(&num_columns,sizeof(int),one,fp); //read n
    printf("initializing M of size %d by %d\n", num_rows, num_columns);
    M = gsl_matrix_alloc(num_rows,num_columns);
    printf("done..\n");

    // read and set elements
    for(i=0; i<num_rows; i++){
        for(j=0; j<num_columns; j++){
            fread(&nnz_val,sizeof(double),one,fp); //read nnz
            gsl_matrix_set(M,i,j,nnz_val);
        }
    }
    fclose(fp);

    return M;
}

/* write matrix to binary file 
 * the nonzeros are in order of double loop over rows and columns
format:
num_rows (int) 
num_columns (int)
nnz (double)
...
nnz (double)
*/
void gsl_matrix_write_to_binary_file(gsl_matrix *M, char *fname) {
    int i, j, num_rows, num_columns, row_num, col_num;
    double nnz_val;
    size_t one = 1;
    FILE *fp;
    num_rows = M->size1; num_columns = M->size2;
    
    fp = fopen(fname,"w");
    fwrite(&num_rows,sizeof(int),one,fp); //write m
    fwrite(&num_columns,sizeof(int),one,fp); //write n

    // write the elements
    for(i=0; i<num_rows; i++){
        for(j=0; j<num_columns; j++){
            nnz_val = gsl_matrix_get(M,i,j);
            fwrite(&nnz_val,sizeof(double),one,fp); //write nnz
        }
    }
    fclose(fp);
}

/* load vector from file 
format:
num_rows
value
.....
value
*/
gsl_vector * gsl_vector_load_from_text_file(char *fname) {
    int i, j, num_rows;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    gsl_vector *v;
    
    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read dimension 
    sscanf(line, "%d", &num_rows);
    v = gsl_vector_calloc(num_rows);

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<num_rows; i++){
        fgets(line,100,fp); 
        sscanf(line, "%s", nnz_val_str);
        nnz_val = atof(nnz_val_str);
        gsl_vector_set(v, i, nnz_val);
    }
    fclose(fp);

    // clean
    free(line);
    free(nnz_val_str);

    return v;
}

/* frobenius norm */
double gsl_get_matrix_frobenius_norm(gsl_matrix *M){
    int i,j;
    double val, norm = 0;
    for(i=0; i<M->size1; i++){
        for(j=0; j<M->size2; j++){
            val = gsl_matrix_get(M, i, j);
            norm += val*val;
        }
    }
    norm = sqrt(norm);
    return norm;
}


/* write sparse matrix to binary file assuming row compressed format */ 
void gsl_sparse_matrix_write_to_binary_file(gsl_spmatrix *A, char *fname){
    FILE *fp;
    int i,m,n,nnz,intval;
    double dblval;
    size_t one = 1;
    nnz = A->nz;
    m = A->size1;
    n = A->size2;

    fp = fopen(fname,"w");

    printf("writing m = %d, n = %d, nnz = %d\n", m,n,nnz);
    fwrite(&m,sizeof(int),one,fp); //write m
    fwrite(&n,sizeof(int),one,fp); //write n
    fwrite(&nnz,sizeof(int),one,fp); //write nnz

    for (i = 0; i < nnz; i++){
        intval = A->i[i];
        fwrite(&intval,sizeof(int),one,fp);
    }

    for(i = 0; i < (m+1); i++){
        intval = A->p[i];
        fwrite(&intval,sizeof(int),one,fp);
    }

    for(i = 0; i < nnz; i++){
        dblval = A->data[i];
        fwrite(&dblval,sizeof(double),one,fp);
    }

    fclose(fp);
}



/* C = A*B */
void gsl_matrix_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}


/* C = A^T*B */
void gsl_matrix_transpose_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C){
    gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
}


/* y = M*x */
void gsl_matrix_vector_mult(gsl_matrix *M, gsl_vector *x, gsl_vector *y){
    gsl_blas_dgemv (CblasNoTrans, 1.0, M, x, 0.0, y);
}


/* y = M^T*x */
void gsl_matrix_transpose_vector_mult(gsl_matrix *M, gsl_vector *x, gsl_vector *y){
    gsl_blas_dgemv (CblasTrans, 1.0, M, x, 0.0, y);
}


/* compute compact QR factorization 
M is mxn; Q is mxk and R is kxk
*/
void gsl_compute_QR_compact_factorization(gsl_matrix *M, gsl_matrix *Q, gsl_matrix *R){
    int i,j,m,n,k;
    m = M->size1;
    n = M->size2;
    k = min(m,n);

    //printf("QR setup..\n");
    gsl_matrix *QR = gsl_matrix_calloc(M->size1, M->size2); 
    gsl_vector *tau = gsl_vector_alloc(min(M->size1,M->size2));
    gsl_matrix_memcpy (QR, M);

    //printf("QR decomp..\n");
    gsl_linalg_QR_decomp (QR, tau);

    //printf("extract R..\n");
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            if(j>=i){
                gsl_matrix_set(R,i,j,gsl_matrix_get(QR,i,j));
            }
        }
    }

    //printf("extract Q..\n");
    gsl_vector *vj = gsl_vector_calloc(m);
    for(j=0; j<k; j++){
        gsl_vector_set(vj,j,1.0);
        gsl_linalg_QR_Qvec (QR, tau, vj);
        gsl_matrix_set_col(Q,j,vj);
        vj = gsl_vector_calloc(m);
    } 
}

/* write sparse matrix to binary file assuming row compressed format */ 
void gsl_sparse_matrix_write_to_binary_file(gsl_spmatrix *A, char *fname){
    FILE *fp;
    int i,m,n,nnz,intval;
    double dblval;
    size_t one = 1;
    nnz = A->nz;
    m = A->size1;
    n = A->size2;

    fp = fopen(fname,"w");

    printf("writing m = %d, n = %d, nnz = %d\n", m,n,nnz);
    fwrite(&m,sizeof(int),one,fp); //write m
    fwrite(&n,sizeof(int),one,fp); //write n
    fwrite(&nnz,sizeof(int),one,fp); //write nnz

    for (i = 0; i < nnz; i++){
        intval = A->i[i];
        fwrite(&intval,sizeof(int),one,fp);
    }

    for(i = 0; i < (m+1); i++){
        intval = A->p[i];
        fwrite(&intval,sizeof(int),one,fp);
    }

    for(i = 0; i < nnz; i++){
        dblval = A->data[i];
        fwrite(&dblval,sizeof(double),one,fp);
    }

    fclose(fp);
}

/* write sparse matrix to binary file assuming row compressed format */
gsl_spmatrix * gsl_sparse_matrix_load_from_binary_file(char *fname){
    FILE *fp;
    int i,m,n,nnz,intval;
    double dblval;
    size_t one = 1;

    fp = fopen(fname,"r");

    fread(&m,sizeof(int),one,fp); //read m
    fread(&n,sizeof(int),one,fp); //read n
    fread(&nnz,sizeof(int),one,fp); //read nnz
    printf("read m=%d,n=%d,nnz=%d\n",m,n,nnz);

    // initialize matrix in gsl crs sparse format
    gsl_spamatrix *A = gsl_spmatrix_alloc_nzmax (m, n, nnz, GSL_SPMATRIX_CRS);
    // once this is done, A has i, p, data arrays associated with it

    for (i = 0; i < nnz; i++){
        fread(&intval,sizeof(int),one,fp);
        A->i[i] = intval;
    }

    for(i = 0; i < (m+1); i++){
        fread(&intval,sizeof(int),one,fp);
        A->p[i] = intval;
    }

    for(i = 0; i < nnz; i++){
        fread(&dblval,sizeof(double),one,fp);
        A->data[i] = dblval;
    }

    fclose(fp);
    return A;
}

/* load vector from file 
format:
num_rows
value
.....
value
*/
gsl_vector * gsl_vector_load_from_text_file(char *fname){
    int i, j, num_rows;
    double nnz_val;
    char *nnz_val_str;
    char *line;
    FILE *fp;
    gsl_vector *v;
    
    line = (char*)malloc(200*sizeof(char));
    fp = fopen(fname,"r");
    fgets(line,100,fp); //read dimension 
    sscanf(line, "%d", &num_rows);
    v = gsl_vector_calloc(num_rows);

    // read and set elements
    nnz_val_str = (char*)malloc(50*sizeof(char));
    for(i=0; i<num_rows; i++){
        fgets(line,100,fp); 
        sscanf(line, "%s", nnz_val_str);
        nnz_val = atof(nnz_val_str);
        gsl_vector_set(v, i, nnz_val);
    }
    fclose(fp);

    // clean
    free(line);
    free(nnz_val_str);

    return v;
}

/* print a sparse matrix out in dense format */
void gsl_sparse_matrix_print(gsl_spmatrix *A){
    int i,j;
    double val;
    for(i=0; i<A->size1; i++){
        for(j=0; j<A->size2; j++){
            val = gsl_spmatrix_get(A, i, j);
            printf("%f  ", val);
        }
        printf("\n");
    }
}

/* print a column vector */
void gsl_vector_print(gsl_vector *v){
    int i,n = v->size;
    printf("n = %d\n", n);
    printf("[ ");
    for(i=0; i<n; i++){
        printf("%4.2f ", gsl_vector_get(v,i));
    }
    printf("]^T\n");
}

/* frobenius norm of a sparse matrix */
double gsl_get_sparse_matrix_frobenius_norm(gsl_spmatrix *A){
    double val = 0, tmp;
    int i,nz = A->nz;
    for(i=0; i<nz; i++){
        val += (A->data[i])*(A->data[i]);
    }
    return sqrt(val);
}

/*  y \leftarrow \alpha (A) v + \beta y */
void gsl_sparse_matrix_vector_mult(gsl_spmatrix *A,gsl_vector *v, 
    gsl_vector **y){
    *y = gsl_vector_calloc(A->size1);
    gsl_spblas_dgemv(CblasNoTrans, 1.0, A, v, 0.0, *y);
}

/*  y \leftarrow \alpha transpose(A) v + \beta y */
void gsl_sparse_matrix_transpose_vector_mult(gsl_spmatrix *A,gsl_vector *v, gsl_vector **y){
    *y = gsl_vector_calloc(A->size2);
    gsl_spblas_dgemv(CblasTrans, 1.0, A, v, 0.0, *y);
}
