import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from dcbo.bases.tfp_model_wrapper import TFPModelWrapper
from numpy import arange, asarray, hstack, random, repeat
from numpy.core import vstack
from dcbo.bases.root import Root
from dcbo.bayes_opt.intervention_computations import evaluate_acquisition_function
from dcbo.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    make_column_shape_2D,
    standard_mean_function,
    zero_variance_adjustment,
)
from tqdm import trange


class ABO(Root):
    def __init__(
        self,
        G: str,
        sem: classmethod,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        number_of_trials: int,
        base_target_variable: str,
        task: str = "min",
        exploration_sets: list = None,
        cost_type: int = 1,
        n_restart: int = 1,
        hp_i_prior: bool = True,
        debug_mode: bool = False,
        optimal_assigned_blankets: dict = None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        args = {
            "G": G,
            "sem": sem,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": exploration_sets,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "number_of_trials": number_of_trials,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**args)
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        self.hp_i_prior = hp_i_prior
        self.seed = seed
        # Convert observational samples to dict of temporal lists.
        # We do this because at each time-index we may have a different number of samples.
        # Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)
        # This is partiuclar to BO, why these lines override the standards in the Root class.
        for temporal_index in range(self.T):
            self.mean_function[temporal_index][self.exploration_sets[0]] = standard_mean_function
            self.variance_function[temporal_index][self.exploration_sets[0]] = zero_variance_adjustment

    def run(self):

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.T, desc="Time index"):

            if self.debug_mode:
                print("Time:", temporal_index)

            # Evaluate each target
            target = self.all_target_variables[temporal_index]

            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):
                #  This function runs the actual computation -- calls are identical for all methods
                self._per_trial_computations(temporal_index, it, target, assigned_blanket, method="ABO")

            # Post optimisation assignments (post this time-index that is)
            self._post_optimisation_assignments(target, temporal_index)

    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket
        return assigned_blanket

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):
        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None

            # We evaluate this function IF there is interventional data at this time index
            if self.seed_anchor_points is None:
                seed_to_pass = None
            else:
                seed_to_pass = int(self.seed_anchor_points * (temporal_index + 1) * it)
            (self.y_acquired[es], self.corresponding_x[es],) = evaluate_acquisition_function(
                self.intervention_exploration_domain[es],
                bo_model,
                self.mean_function[temporal_index][es],
                self.variance_function[temporal_index][es],
                current_best_global_target,
                es,
                self.cost_functions,
                self.task,
                self.base_target_variable,
                dynamic=True,
                causal_prior=False,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                sample_anchor_points=self.sample_anchor_points,
                seed_anchor_points=seed_to_pass,
            )

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, alpha: float = 2, beta: float = 0.5,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)
        time_indices = arange(0, temporal_index + 1)
        counts = asarray([self.interventional_data_x[t][exploration_set].shape[0] for t in range(temporal_index + 1)])
        time_vec = make_column_shape_2D(repeat(time_indices, counts))

        if temporal_index > 0:
            data_x = vstack([self.interventional_data_x[t][exploration_set] for t in range(temporal_index + 1)])
            data_y = vstack([self.interventional_data_y[t][exploration_set] for t in range(temporal_index + 1)])
        else:
            data_x = self.interventional_data_x[temporal_index][exploration_set]
            data_y = self.interventional_data_y[temporal_index][exploration_set]

        x = hstack((data_x, time_vec))
        y = data_y.squeeze(1)
        # print("inp:", x.shape, y.shape)

        if not self.bo_model[temporal_index][exploration_set]:

            input_dim = input_dim + 1

            observation_noise_variance = tf.constant(1e-5, dtype=tf.float64)
            variance = tfp.util.TransformedVariable(
                initial_value=1.0,
                bijector=tfp.bijectors.Softplus(),
                dtype=tf.float64,
                prior=tfp.distributions.Gamma(concentration=alpha, rate=beta)
            ) if self.hp_i_prior else 1.0

            abo_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=variance, length_scale=1.0)
            gp = tfp.distributions.GaussianProcess(
                kernel=abo_kernel,
                index_points=tf.convert_to_tensor(x, dtype=tf.float64),
                observation_noise_variance=observation_noise_variance,
            )

            old_np_seed = np.random.get_state()
            old_tf_seed = tf.random.get_global_generator().state
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

            optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

            # @tf.function
            def _optimize():
                with tf.GradientTape() as tape:
                    # 计算负对数边缘似然
                    loss = -gp.log_prob(tf.convert_to_tensor(y, dtype=tf.float64))
                gradients = tape.gradient(loss, gp.trainable_variables)
                optimizer.apply_gradients(zip(gradients, gp.trainable_variables))
                return loss
            
            opt_step = 200
            for i in range(opt_step):
                neg_log_likelihood_ = _optimize()
                # if i % 30 == 0:
                #     print("Step {}: NLL = {}".format(i, neg_log_likelihood_))

            np.random.set_state(old_np_seed)
            tf.random.get_global_generator().reset_from_seed(tf.random.experimental.create_rng_state(old_tf_seed, 'threefry'))

            self.bo_model[temporal_index][exploration_set] = TFPModelWrapper(
                kernel=abo_kernel, 
                observation_index_points=tf.convert_to_tensor(x, dtype=tf.float64),
                observations=tf.convert_to_tensor(y, dtype=tf.float64),
                observation_noise_variance=observation_noise_variance,
                optimizer = optimizer,
                opt_step = opt_step
            )
        else:
            # Model exists so we simply update it
            self.bo_model[temporal_index][exploration_set].set_data(
                X=tf.convert_to_tensor(x, dtype=tf.float64),
                Y=tf.convert_to_tensor(y, dtype=tf.float64),
            )
            
            old_np_seed = np.random.get_state()
            old_tf_seed = tf.random.get_global_generator().state
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

            self.bo_model[temporal_index][exploration_set].optimize()

            np.random.set_state(old_np_seed)
            tf.random.get_global_generator().reset_from_seed(tf.random.experimental.create_rng_state(old_tf_seed, 'threefry'))
