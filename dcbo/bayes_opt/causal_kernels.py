import tensorflow as tf
import tensorflow_probability as tfp

class CausalRBF(tfp.math.psd_kernels.PositiveSemidefiniteKernel):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel.
    """

    def __init__(
        self,
        variance_adjustment,
        variance=1.0,
        lengthscale=None,
        rescale_variance=1.0,
        ARD=False,
        active_dims=None,
        name="rbf",
        useGPU=False,
        inv_l=False,
        dtype=tf.float64,
        feature_ndims=1,
    ):
        parameters = dict(locals())
        with tf.name_scope(name):
            # Define variance parameter with Softplus transformation to ensure positivity
            self.variance = tfp.util.TransformedVariable(
                initial_value=variance * rescale_variance,
                bijector=tfp.bijectors.Softplus(),
                dtype=dtype,
                name='variance'
            )
            
            # Adjust variance with variance_adjustment
            self.variance_adjustment = variance_adjustment
            
            # Define lengthscale parameter with Exp transformation to ensure positivity
            if lengthscale is None:
                lengthscale = 1.0  # Default value if not provided
                
            if ARD and isinstance(lengthscale, float):
                # For ARD, we expect a list of lengthscales for each dimension
                raise ValueError("For ARD, lengthscale must be a list or array.")
                
            if inv_l:
                # If using inverse lengthscale, apply reciprocal transformation
                self.lengthscale = tfp.util.TransformedVariable(
                    initial_value=lengthscale,
                    bijector=tfp.bijectors.Reciprocal(),
                    dtype=dtype,
                    name='inv_lengthscale'
                )
            else:
                self.lengthscale = tfp.util.TransformedVariable(
                    initial_value=lengthscale,
                    bijector=tfp.bijectors.Exp(),
                    dtype=dtype,
                    name='lengthscale'
                )

            self.ARD = ARD
            self.active_dims = active_dims
            self.useGPU = useGPU

        super(CausalRBF, self).__init__(feature_ndims, dtype, name, parameters)

    def _batch_shape(self):
        return tf.TensorShape([])

    # @tf.function
    def _apply(self, X, X2=None, example_ndims=0):
        # print("X2X shape:", X.shape, X2.shape)
        
        if X2 is None:
            X2 = X
                
        r2 = self._scaled_dist(X, X2)  # remove the sqrt operator
        values = self.variance * tf.exp(-0.5 * r2)

        # Ensure variance_adjustment returns a TensorFlow tensor
        # import pdb
        # pdb.set_trace()

        value_diagonal_X = self.variance_adjustment(X.numpy())
        value_diagonal_X2 = self.variance_adjustment(X2.numpy())
        value_diagonal_X = tf.convert_to_tensor(value_diagonal_X)
        value_diagonal_X2 = tf.convert_to_tensor(value_diagonal_X2)
        # print("[kernel val_diagonal]:", value_diagonal_X.shape, value_diagonal_X2.shape)

        additional_matrix = tf.matmul(tf.sqrt(value_diagonal_X), tf.sqrt(value_diagonal_X2), transpose_b=True)
        
        # import pdb
        # pdb.set_trace()
        # print("[kernel end]:", additional_matrix.shape, values.shape, X.shape, X2.shape)
        # Use TensorFlow debugging assertions instead of Python's assert
        tf.debugging.assert_equal(
            tf.shape(additional_matrix),
            tf.shape(values),
            message=f"Shapes do not match: {tf.shape(additional_matrix)} vs {tf.shape(values)}"
        )
        return values + additional_matrix

    def _scaled_dist(self, X, X2=None):
        if self.ARD:
            if X2 is not None:
                X2 = X2 / self.lengthscale
            return self._unscaled_dist(X/self.lengthscale, X2)
        else:
            return self._unscaled_dist(X, X2)/self.lengthscale

    def _unscaled_dist(self, X, X2=None):
        if X2 is None:
            X2 = X
        X1sq = tf.reduce_sum(tf.square(X), axis=1)
        X2sq = tf.reduce_sum(tf.square(X2), axis=1)
        dot_product = tf.matmul(X, X2, transpose_b=True)
        r2 = -2. * dot_product + tf.expand_dims(X1sq, axis=1) + tf.expand_dims(X2sq, axis=0)
        # 对差值进行平方
        r2 = tf.square(r2)
        return r2

    def _matrix(self, x1, x2):
        return self._apply(x1, x2)

    @property
    def trainable_variables(self):
        return self.variance.trainable_variables + self.lengthscale.trainable_variables

    @property
    def variables(self):
        return self.variance.variables + self.lengthscale.variables
    
# Example usage:
# causal_rbf = CausalRBF(variance_adjustment=1.0, variance=1.0, lengthscale=[1.0, 2.0], ARD=True)

import numpy as np

class CustomMeanFunction:
    def __init__(self, mean_function, update_gradients):
        self.mean_function = mean_function
        self.update_gradients = update_gradients

    def __call__(self, x):
        # 假设 mean_function 返回的是一个 numpy 数组或可以转换为 tensor 的其他类型
        if isinstance(x, np.ndarray):
            x_np = x
        else:
            x_np = x.numpy()
        
        # 调用原始的 mean_function 并获取结果
        result = self.mean_function(x_np).squeeze(1)
        # print("custom:", result.shape)
        # 将结果转换为 Tensor 并确保其形状正确
        return tf.convert_to_tensor(result, dtype=tf.float64)
