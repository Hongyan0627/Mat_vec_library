#include "mat_vec.h"

int main() {
	mat *M;
	vec *x,*y;
	double vecnorm;

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

	y = matrix_vec_mult(M,x);

	printf("\t equals\n");
	vector_print(y);

	printf("next, test loading from binary files..\n");
	matrix_delete(M);
	vector_delete(x);
	vector_delete(y);

	//printf("%s\n", matrix_location_bin_file);
	//M = matrix_load_from_binary_file(matrix_location_bin_file);
	M = matrix_load_from_binary_file("data/A_mat1.bin");
	x = vector_load_from_binary_file("data/x1.bin");



	y = matrix_vec_mult(M,x);
	vecnorm = get_vector_euclidean_norm(y);
	printf("norm(y) = %f\n", vecnorm);		

	return 0;
}