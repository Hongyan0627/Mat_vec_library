#include "mat_vec.h"

int main() {
	mat *M;
	vec *x,*y;

	printf("first, hand set inputs..\n");
	M = matrix_new(2,2);	
	matrix_set_element(M,0,0,2);
	matrix_set_element(M,1,1,3);	
	matrix_print(M);

	x = vector_new(2);
	vector_set_element(x,0,3);
	vector_set_element(x,1,4);
	printf("\t times \n");
	vector_print(x);

	printf("next, test loading from binary files..\n");
	matrix_delete(M);
	vector_delete(x);

	gsl_matrix *Q,*R;

	Q = gsl_matrix_alloc(5,5);
	R = gsl_matrix_alloc(5,5);

	return 0;
}