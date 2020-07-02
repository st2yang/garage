from garage.torch.algos import VPG


class A2C(VPG):
    """
    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

        Args:
            env_spec (garage.envs.EnvSpec): Environment specification.
            policy (garage.torch.policies.Policy): Policy.
            value_function (garage.torch.value_functions.ValueFunction): The value
                function.
            policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
                for policy.
            vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
                value function.
            max_path_length (int): Maximum length of a single rollout.
            num_train_per_epoch (int): Number of train_once calls per epoch.
            discount (float): Discount.
            gae_lambda (float): Lambda used for generalized advantage
                estimation.
            center_adv (bool): Whether to rescale the advantages
                so that they have mean 0 and standard deviation 1.
            positive_adv (bool): Whether to shift the advantages
                so that they are always positive. When used in
                conjunction with center_adv the advantages will be
                standardized before shifting.
            policy_ent_coeff (float): The coefficient of the policy entropy.
                Setting it to zero would mean no entropy regularization.
            use_softplus_entropy (bool): Whether to estimate the softmax
                distribution of the entropy to prevent the entropy from being
                negative.
            stop_entropy_gradient (bool): Whether to stop the entropy gradient.
            entropy_method (str): A string from: 'max', 'regularized',
                'no_entropy'. The type of entropy method to use. 'max' adds the
                dense entropy to the reward for each time step. 'regularized' adds
                the mean entropy to the surrogate objective. See
                https://arxiv.org/abs/1805.00909 for more details.

        """

    def __init__(
            self,
            env_spec,
            policy,
            value_function,
            policy_optimizer=None,
            vf_optimizer=None,
            max_path_length=500,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            entropy_method='regularized',
    ):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         max_path_length=max_path_length,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)

    def process_samples(self, paths):
        r"""Process sample data based on the collected paths.

        This method bootstraps each paths's final state's reward
        fromt the value function and discounts to find the remaining
        rewards. see https://arxiv.org/pdf/1602.01783.pdf

        Notes: P is the maximum path length (self.max_path_length)

        Args:
            paths (list[dict]): A list of collected paths

        Returns:
            torch.Tensor: The observations of the environment
                with shape :math:`(N, P, O*)`.
            torch.Tensor: The actions fed to the environment
                with shape :math:`(N, P, A*)`.
            torch.Tensor: The acquired rewards with shape :math:`(N, P)`.
            list[int]: Numbers of valid steps in each paths.
            torch.Tensor: Value function estimation at each step
                with shape :math:`(N, P)`.

        """
        # bootstrap from value function
        for path in paths:
            if path['dones'][-1]:
                path['rewards'][-1] = 0
            else:
                path['rewards'][-1] = self.value_function.forward(
                    [path['observations'][-1]])
        return super().process_samples(paths)
