

/**
 * Desired codegeneration for A.dot(v):
 *
 *  A: n x n square matrix
 *  v: n x 1 vector
 */
void matrix_vector (double *A, double *v, double *result) {
  for (int i=0; i<$n; i++) {
    double sum = 0.0;
    for (int j=0; j<$n; j++) {
      sum += A[i * $n + j] * v[j];
    }
    result[i] = sum;
  }
}

/**
 * Desired codegeneration for v.T.dot(a)
 *
 *  v: n x 1 vector
 *  a: n x 1 vector
 */
void vector_vector (double *v, double *a, double *result) {
  double sum = 0.0;
  for (int i=0; i<$n; i++) {
    sum += v[i] * a[i];
  }
  *result = sum
}

/**
 * Desired codegeneration for v.T.dot(A).dot(v)
 *
 *  A: n x n vector
 *  v: n x 1 vector
 */
void vector_matrix_vector (double *A, double *v, double *result) {
   for (int i=0; i<$n; i++) {
    double sum = 0.0;
    for (int j=0; j<$n; j++) {
      sum += A[i * $n + j] * v[j];
    }
    *result += v[i] * sum;
  }
}