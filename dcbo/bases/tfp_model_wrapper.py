from emukit.core.interfaces import IModel, IDifferentiable
import tensorflow as tf
import tensorflow_probability as tfp

dtype = tf.float64

class TFPModelWrapper(IModel, IDifferentiable):
    def __init__(self, kernel, 
                observation_index_points,
                observations,
                observation_noise_variance=tf.constant(1e-6, dtype=dtype),
                mean_fn=None,
                optimizer=None,
                opt_step=200):
        
        self.kernel = kernel
        self.observation_index_points = observation_index_points
        self.observations = observations
        self.observation_noise_variance = observation_noise_variance
        self.mean_fn = mean_fn

        self.optimizer = optimizer if optimizer is not None else tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)
        self.opt_step = opt_step


    def predict(self, X_test):
        # 使用 TFP 模型进行预测
        # print("X_test:", X_test.shape)
        gprm = tfp.distributions.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=tf.convert_to_tensor(X_test, dtype=dtype),
            observation_index_points=tf.convert_to_tensor(self.observation_index_points, dtype),
            observations=tf.convert_to_tensor(self.observations, dtype),
            observation_noise_variance=self.observation_noise_variance,
            mean_fn=self.mean_fn, 
            jitter=1e-6
        )
        # import pdb
        # pdb.set_trace()

        mean = gprm.mean().numpy()
        variance = gprm.stddev().numpy()
        if len(variance.shape) == 2:
            variance = variance.diagonal()    
        
        return mean, variance

    def set_data(self, X, Y):
        self.observation_index_points = X
        self.observations = Y

    def get_prediction_gradients(self, X):
        # 计算预测值的梯度
        with tf.GradientTape() as tape:
            tape.watch(X)
            predictive_distribution = self.model(X)
            mean = predictive_distribution.mean()
        gradients = tape.gradient(mean, X).numpy()
        return mean.numpy(), gradients

    def optimize(self):
        # 实现模型优化逻辑
        gp = tfp.distributions.GaussianProcess(
                kernel=self.kernel,
                index_points=tf.convert_to_tensor(self.observation_index_points, dtype=tf.float64),
                observation_noise_variance=self.observation_noise_variance
            )
        
        def _optimize():
            with tf.GradientTape() as tape:
                # 计算负对数边缘似然
                loss = -gp.log_prob(tf.convert_to_tensor(self.observations, dtype=tf.float64))
            gradients = tape.gradient(loss, self.kernel.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.kernel.trainable_variables))
            return loss
        
        for i in range(self.opt_step):
            neg_log_likelihood_ = _optimize()
            # if i % 30 == 0:
            #     print("Step {}: NLL = {}".format(i, neg_log_likelihood_))