import torch
import numpy as np


class DDPMSampler:

    def __init__(
        self,
        random_generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        """
        Initializes a diffusion process object with parameters for beta scheduling
        and timestep configuration, following the DDPM (Denoising Diffusion Probabilistic Models) framework.

        Args:
            random_generator (torch.Generator): A PyTorch random number generator for reproducibility.
            num_training_steps (int, optional): The total number of training timesteps in the diffusion process. Defaults to 1000.
            beta_start (float, optional): The starting value of the beta schedule. Defaults to 0.00085.
            beta_end (float, optional): The ending value of the beta schedule. Defaults to 0.0120.

        Attributes:
            betas (torch.Tensor): A tensor of beta values representing the noise schedule.
                                Shape: (num_training_steps,).
            alphas (torch.Tensor): A tensor of alpha values (1 - beta).
                                Shape: (num_training_steps,).
            alphas_cumprod (torch.Tensor): The cumulative product of alpha values, used for scaling during the diffusion process.
                                        Shape: (num_training_steps,).
            one (torch.Tensor): A tensor with a constant value of 1.0 for scaling purposes.
            random_generator (torch.Generator): The random number generator for reproducibility.
            num_training_steps (int): The total number of training timesteps.
            timesteps (torch.Tensor): A tensor containing the reversed sequence of timesteps.
                                    Shape: (num_training_steps,).

        References:
            - Beta schedule values are taken from:
            https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
            - Naming conventions align with the DDPM paper:
            https://arxiv.org/pdf/2006.11239.pdf

        Example:
            >>> generator = torch.Generator().manual_seed(42)
            >>> diffusion_process = DiffusionProcess(generator, num_training_steps=1000)
            >>> print(diffusion_process.betas.shape)
            torch.Size([1000])
        """
        # Generate the beta schedule with a squared-root scaling
        self.beta_schedule = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )

        # Compute alpha values (1 - beta) based on paper formula
        self.alpha_schedule = 1.0 - self.beta_schedule

        # Compute cumulative product of alpha values [alpha_0, alpha_1 * alpha_0 ...]
        self.alpha_cumulative_product = torch.cumprod(self.alpha_schedule, dim=0)

        # Define a constant tensor with the value 1.0
        self.constant_one = torch.tensor(1.0)

        # Assign the provided random number generator
        self.random_generator = random_generator

        # Store the total number of training timesteps
        self.total_timesteps = num_training_steps

        # Create a reversed sequence of timesteps, we want to go from all noise to no noise
        self.reversed_timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy()
        )

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        """
        Configures the number of inference timesteps for the diffusion process and calculates
        the corresponding timesteps based on the total training timesteps.

        Args:
            num_inference_steps (int, optional): The number of timesteps to use during inference. Defaults to 50.

        Attributes Updated:
            num_inference_steps (int): The number of inference timesteps configured.
            inference_timesteps (torch.Tensor): A tensor containing the calculated timesteps for inference,
                                                scaled and reversed based on the training timesteps.
                                                Shape: (num_inference_steps,).

        Calculation:
            The method divides the total number of training timesteps (`total_training_timesteps`) by
            the desired number of inference timesteps (`num_inference_steps`) to calculate a step ratio.
            Timesteps are then spaced evenly using this ratio and reversed for the inference process.

        Example:
            >>> diffusion_process.set_inference_timesteps(num_inference_steps=100)
            >>> print(diffusion_process.inference_timesteps)
            tensor([999, 989, 979, ..., 19, 9, 0], dtype=torch.int64)
        """
        # Store the number of inference timesteps
        self.num_inference_steps = num_inference_steps

        # Calculate the ratio of training steps to inference steps
        step_ratio = self.total_timesteps // self.num_inference_steps

        # Generate evenly spaced timesteps, scale them, reverse their order, and convert to integer type
        inference_timestep_indices = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .astype(np.int64)
        )

        # Convert the numpy array of timesteps into a PyTorch tensor
        self.reversed_timesteps = torch.from_numpy(inference_timestep_indices)

    def add_noise(
        self,
        clean_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Adds noise to input samples according to the diffusion process, based on given timesteps.

        Args:
            clean_samples (torch.FloatTensor): The original clean input samples with shape (batch_size, ...).
            timesteps (torch.IntTensor): The timesteps for which noise is to be added, with shape (batch_size,).

        Returns:
            torch.FloatTensor: Noisy samples generated from the input clean samples.
        """
        # Move cumulative product of alphas to the same device and type as the input samples
        alpha_cumulative_products = self.alpha_cumulative_product.to(
            device=clean_samples.device, dtype=clean_samples.dtype
        )
        timesteps = timesteps.to(clean_samples.device)

        # Calculate the square root of alpha cumulative products for the given timesteps referenc paper for equation
        sqrt_alpha_cumprod = alpha_cumulative_products[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        # to match dimension just keep adding it until match with clean_sample tensor shape using unsqueese
        while len(sqrt_alpha_cumprod.shape) < len(clean_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

        # Calculate the square root of (1 - alpha cumulative products) for the given timesteps - paper
        # std, see encoder how to sample from distibution
        sqrt_one_minus_alpha_cumprod = (
            1 - alpha_cumulative_products[timesteps]
        ) ** 0.5  # standard deiation
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(clean_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        # Generate noise from a standard normal distribution
        noise = torch.randn(
            clean_samples.shape,
            generator=self.random_generator,
            device=clean_samples.device,
            dtype=clean_samples.dtype,
        )

        # Add noise to the clean samples using the diffusion process equation
        # formula (4) from ddpm paper
        noisy_samples = (
            sqrt_alpha_cumprod * clean_samples + sqrt_one_minus_alpha_cumprod * noise
        )

        return noisy_samples

    def _get_previous_timestep(self, current_timestep: int) -> int:
        """
        Calculates the previous timestep index during the reverse diffusion process.

        Args:
            current_timestep (int): The current timestep `t` in the diffusion process.

        Returns:
            int: The previous timestep index `t-1`. This is determined based on the number
            of training timesteps and inference steps.
        """
        # Compute the step size between inference timesteps
        timestep_step_size = self.total_timesteps // self.num_inference_steps

        # Calculate the previous timestep index
        previous_timestep = current_timestep - timestep_step_size

        return previous_timestep

    def _get_variance(self, current_timestep: int) -> torch.Tensor:
        """
        Computes the variance for the reverse diffusion process at a given timestep.

        Args:
            current_timestep (int): The current timestep `t` in the reverse diffusion process.

        Returns:
            torch.Tensor: The variance value for the current timestep. This value is
            clamped to avoid numerical instability (e.g., log of zero).

        Notes:
            - The variance computation is based on equations (6) and (7) from the
            DDPM paper: https://arxiv.org/pdf/2006.11239.pdf.
            - Variance is used to sample from the Gaussian distribution to predict
            the previous sample `x_{t-1}` during the reverse process.
        """
        # Compute the previous timestep index
        previous_timestep = self._get_previous_timestep(current_timestep)

        # Retrieve cumulative alpha values for the current and previous timesteps
        alpha_cumulative_product_t = self.alpha_cumulative_product[current_timestep]
        alpha_cumulative_product_previous = (
            self.alpha_cumulative_product[previous_timestep]
            if previous_timestep >= 0
            else self.one
        )

        # Compute the current beta value
        current_beta_t = 1 - (
            alpha_cumulative_product_t / alpha_cumulative_product_previous
        )

        # Compute the variance as per formula (6) and (7) in the DDPM paper
        variance = (
            (1 - alpha_cumulative_product_previous) / (1 - alpha_cumulative_product_t)
        ) * current_beta_t

        # Clamp variance to avoid numerical instability (e.g., log of 0)
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, noise_strength: float = 1.0) -> None:
        """
        Configures the noise level to add to the input image during the diffusion process.

        Args:
            noise_strength (float, optional): The degree of noise to add to the input image.
                - A value of `1.0` adds the maximum amount of noise, making the output less similar to the input image.
                - A value of `0.0` adds minimal noise, making the output closely resemble the input image.
                Default is `1.0`.

        Updates:
            - Adjusts the timesteps to skip initial noise levels based on the provided strength.
            - Computes and stores the number of timesteps skipped as `self.start_step`.

        Notes:
            - `noise_strength` determines how far the output deviates from the input.
            - Higher strength values lead to more creative outputs but with less resemblance to the input.
        """
        # Compute the starting step based on the noise strength
        start_step_index = self.num_inference_steps - int(
            self.num_inference_steps * noise_strength
        )

        # Adjust the timesteps to skip the initial steps
        self.reversed_timesteps = self.reversed_timesteps[start_step_index:]

        # Store the starting step index
        self.start_step = start_step_index

    def step(
        self,
        current_timestep: int,
        latent_samples: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the predicted latent sample for the previous timestep during the reverse diffusion process.

        Args:
            current_timestep (int): The current timestep `t` in the diffusion process.
            latent_samples (torch.Tensor): The latent samples at the current timestep `t`, with shape (batch_size, ...).
            predicted_noise (torch.Tensor): The noise predicted by the model for the current timestep, with shape (batch_size, ...).

        Returns:
            torch.Tensor: The predicted latent samples for the previous timestep `t-1`, with shape (batch_size, ...).
        """
        # Compute the previous timestep index
        previous_timestep = self._get_previous_timestep(current_timestep)

        # 1. Compute alpha and beta values
        alpha_product_current = self.alpha_cumulative_product[current_timestep]
        alpha_product_previous = (
            self.alpha_cumulative_product[previous_timestep]
            if previous_timestep >= 0
            else self.constant_one
        )  # TODO: could be improved

        # HuggingFace reference
        beta_product_current = 1 - alpha_product_current
        beta_product_previous = 1 - alpha_product_previous
        alpha_current_t = alpha_product_current / alpha_product_previous
        beta_current_t = 1 - alpha_current_t

        # 2. Predict the original sample (x_0) using formula (15) from DDPM paper
        predicted_original_sample = (
            latent_samples - beta_product_current**0.5 * predicted_noise
        ) / alpha_product_current**0.5

        # 3. Compute coefficients for predicted_original_sample (x_0) and current_sample (x_t)
        # Formula (7) from DDPM paper
        original_sample_coefficient = (
            alpha_product_previous**0.5 * beta_current_t
        ) / beta_product_current
        current_sample_coefficient = (
            alpha_current_t**0.5 * beta_product_previous
        ) / beta_product_current

        # 4. Compute predicted sample for the previous timestep µ_t-mean
        predicted_previous_sample = (
            original_sample_coefficient * predicted_original_sample
            + current_sample_coefficient * latent_samples
        )

        # 5. Add noise if not the final timestep
        variance = 0
        if current_timestep > 0:
            device = predicted_noise.device
            noise = torch.randn(
                predicted_noise.shape,
                generator=self.random_generator,
                device=device,
                dtype=predicted_noise.dtype,
            )
            # Compute variance using formula (7) from DDPM paper
            variance = (self._get_variance(current_timestep) ** 0.5) * noise

        # Add the noise component to the predicted previous sample
        predicted_previous_sample += variance

        return predicted_previous_sample
