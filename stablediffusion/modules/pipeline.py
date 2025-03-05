import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from stablediffusion.models.ddpm import DDPMSampler
from stablediffusion.utils.timembedding_fun import *

import matplotlib.pyplot as plt
import taichi as ti

# Defined by production trained models
WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


class Pipeline:
    def __init__(self, models, tokenizer, device="gpu", idle_device="cpu"):
        """_summary_
        models : dict
            A dictionary containing preloaded models, including "clip", "encoder", "diffusion", and "decoder".
        device : str, optional
            The device to run computations on, e.g., "cuda" or "cpu" (default is None).
        idle_device : str, optional
            A secondary device to offload models when they are not in use, to save memory (default is None).
        tokenizer : PreTrainedTokenizer, optional
            The tokenizer used to preprocess the text prompt into tokens (default is None).
        """
        self.models = models
        self.tokenizer = tokenizer
        self.device = device
        self.idle_device = idle_device

        # self.model = self.load_model()
        # self.tokenizer = self.load_tokenizer()

    def generate(
        self,
        prompt,
        uncond_prompt=None,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        seed=None,
    ):
        """
        Generates an image based on a text prompt using a diffusion model.

        Parameters:
        ----------
        prompt : str
            The main text prompt to guide the image generation process.
        uncond_prompt : str, optional
            The unconditioned text prompt used for classifier-free guidance (default is None)-Negativ prompt.
        input_image : PIL.Image.Image, optional
            An input image to guide the generation process. If provided, the generation will be based on this image.
        strength : float, optional
            The degree of influence of the input image during the diffusion process. Must be between 0 and 1 (default is 0.8).
        do_cfg : bool, optional
            Whether to use classifier-free guidance (default is True).
        cfg_scale : float, optional
            The scale for classifier-free guidance to control the trade-off between prompt adherence and creativity (default is 7.5).
        sampler_name : str, optional
            The name of the sampling algorithm to use. Currently supports "ddpm" (default is "ddpm").
        n_inference_steps : int, optional
            The number of diffusion steps for image generation. Higher values improve quality but increase computation time (default is 50).
        models : dict
            A dictionary containing preloaded models, including "clip", "encoder", "diffusion", and "decoder".
        seed : int, optional
            A random seed for reproducibility. If None, a random seed is used (default is None).
        device : str, optional
            The device to run computations on, e.g., "cuda" or "cpu" (default is None).
        idle_device : str, optional
            A secondary device to offload models when they are not in use, to save memory (default is None).
        tokenizer : PreTrainedTokenizer, optional
            The tokenizer used to preprocess the text prompt into tokens (default is None).

        Returns:
        -------
        np.ndarray
            The generated image as a NumPy array with shape (Height, Width, Channels) in the range [0, 255].

        Raises:
        ------
        ValueError
            If `strength` is not between 0 and 1.
            If `sampler_name` is not recognized.

        Example:
        --------
        ```
        generated_image = generate(
            prompt="A futuristic cityscape at sunset",
            models=models,
            device="cuda",
            tokenizer=tokenizer
        )
        ```
        """
        with torch.no_grad():  # Ensure no gradients are calculated to save memory and computation
            if not 0 < strength <= 1:
                raise ValueError("strength must be between 0 and 1")

            # Move tensors to idle device if specified, otherwise keep them on the current device
            if self.idle_device:
                to_idle = lambda x: x.to(self.idle_device)
            else:
                to_idle = lambda x: x

            # Initialize random number generator
            generator = torch.Generator(device=self.device)
            if seed is None:
                generator.seed()  # Use random seed if not provided
            else:
                generator.manual_seed(seed)  # Use provided seed for reproducibility

            # Load the CLIP model for text processing
            clip = self.models["clip"]
            clip.to(self.device)

            if do_cfg:  # Classifier-free guidance setup
                # Tokenize the conditioned prompt and prepare its embeddings, see clip.py
                cond_tokens = self.tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                cond_tokens = torch.tensor(
                    cond_tokens, dtype=torch.long, device=self.device
                )
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                cond_context = clip(
                    cond_tokens
                )  # Get embeddings for the conditioned prompt

                # Tokenize the unconditioned prompt and prepare its embeddings
                # Convert into a list of length Seq_Len=77
                uncond_tokens = self.tokenizer.batch_encode_plus(
                    [uncond_prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(
                    uncond_tokens, dtype=torch.long, device=self.device
                )
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip(
                    uncond_tokens
                )  # Get embeddings for the unconditioned prompt

                # Concatenate conditioned and unconditioned contexts for guidance
                # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
                context = torch.cat([cond_context, uncond_context])
            else:  # No classifier-free guidance
                tokens = self.tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                context = clip(tokens)  # Get embeddings for the single prompt

            to_idle(clip)  # Free up memory by moving CLIP model to idle device

            # Initialize sampler
            if sampler_name == "ddpm":
                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(n_inference_steps)
            else:
                raise ValueError(f"Unknown sampler value {sampler_name}.")

            # Define the shape of the latent space
            latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

            if input_image:  # Use input image for latent initialization
                encoder = self.models["encoder"]
                encoder.to(self.device)

                # Preprocess input image
                input_image_tensor = input_image.resize((WIDTH, HEIGHT))  # Resize image
                # (Height, Width, Channel)
                input_image_tensor = np.array(
                    input_image_tensor
                )  # Convert to numpy array
                # (Height, Width, Channel) -> (Height, Width, Channel)
                input_image_tensor = torch.tensor(
                    input_image_tensor, dtype=torch.float32, device=self.device
                )
                # (Height, Width, Channel) -> (Height, Width, Channel)
                input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
                # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel) unsqueezed
                # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width) permuted for VAE encoder see encoder
                input_image_tensor = input_image_tensor.unsqueeze(0).permute(
                    0, 3, 1, 2
                )  # Rearrange axes

                # Encode the input image into latents
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                encoder_noise = torch.randn(
                    latents_shape, generator=generator, device=self.device
                )
                latents = encoder(input_image_tensor, encoder_noise)

                # Add noise to the encoded latents based on the strength
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                sampler.set_strength(strength=strength)
                latents = sampler.add_noise(latents, sampler.reversed_timesteps[0])

                to_idle(encoder)  # Free up memory by moving encoder to idle device
            else:  # Generate random latents
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                # if we are doing test with text-to-image, start with random noise
                latents = torch.randn(
                    latents_shape, generator=generator, device=self.device
                )

            # Load the diffusion model
            diffusion = self.models["diffusion"]
            diffusion.to(self.device)

            # Iteratively denoise the latents
            timesteps = tqdm(sampler.reversed_timesteps)  # Progress bar for timesteps
            for i, timestep in enumerate(timesteps):
                # Compute time embedding for the current timestep, (1, 320)
                # convert number into vector
                time_embedding = get_time_embedding(timestep).to(self.device)
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = latents
                if do_cfg:  # Duplicate inputs for classifier-free guidance
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                    model_input = model_input.repeat(2, 1, 1, 1)

                # Predict noise using the diffusion model
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                model_output = diffusion(model_input, context, time_embedding)
                if do_cfg:  # Apply classifier-free guidance
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = (
                        cfg_scale * (output_cond - output_uncond) + output_uncond
                    )

                # Update latents by removing predicted noise
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                latents = sampler.step(timestep, latents, model_output)

            to_idle(
                diffusion
            )  # Free up memory by moving diffusion model to idle device

            # Decode the final latents into images
            decoder = self.models["decoder"]
            decoder.to(self.device)
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
            images = decoder(latents)
            to_idle(decoder)  # Free up memory by moving decoder to idle device

            # Post-process the generated images
            images = rescale(images, (-1, 1), (0, 255), clamp_values=True)
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
            images = images.permute(0, 2, 3, 1)  # Rearrange axes for image format
            images = images.to(
                "cpu", torch.uint8
            ).numpy()  # Convert to numpy for output
            return images[0]  # Return the first image

    def _do_cfg_and_tokenize_text(
        self, clip, prompt, uncond_prompt="", do_cfg=True, batch_size=1
    ):
        if do_cfg:  # Classifier-free guidance setup
            # Tokenize the conditioned prompt and prepare its embeddings, see clip.py
            # prompt[0] - pass through batch_encode_plus

            cond_tokens = self.tokenizer.batch_encode_plus(
                prompt, padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(
                cond_tokens, dtype=torch.long, device=self.device
            )
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(
                cond_tokens
            )  # Get embeddings for the conditioned prompt

            # Tokenize the unconditioned prompt and prepare its embeddings
            # Convert into a list of length Seq_Len=77
            u_p = [uncond_prompt] * batch_size
            uncond_tokens = self.tokenizer.batch_encode_plus(
                u_p,
                padding="max_length",
                max_length=77,
                truncation=True,
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(
                uncond_tokens, dtype=torch.long, device=self.device
            )
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(
                uncond_tokens
            )  # Get embeddings for the unconditioned prompt

            # Concatenate conditioned and unconditioned contexts for guidance
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:  # No classifier-free guidance
            tokens = self.tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)  # Get embeddings for the single prompt

        return context

    def train(
        self,
        hyperparameters,
        train_loader,
        sampler_name,
        seed,
        display=True,
        do_cfg=True,
    ):

        # Set diffusion model
        diffusion = self.models["diffusion"]
        diffusion.to(self.device)
        # Use Adam optimizer for diffusion model
        optimizer = optim.Adam(
            diffusion.parameters(), lr=hyperparameters["learning_rate"]
        )

        # Initialize random number generator
        generator = torch.Generator(device=self.device)
        if seed is None:
            generator.seed()  # Use random seed if not provided
        else:
            generator.manual_seed(seed)  # Use provided seed for reproducibility

        # Initialize ddpm sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(hyperparameters["num_inference_steps"])
        else:
            raise ValueError(f"Unknown sampler value {sampler_name}.")

        # Load the CLIP model for text processing
        clip = self.models["clip"]
        clip.to(self.device)

        # Load the VAE model for text processing
        encoder = self.models["encoder"]
        encoder.to(self.device)

        # Decode the final latents into images
        decoder = self.models["decoder"]
        decoder.to(self.device)

        loss_fn = nn.MSELoss()

        tb_writer = SummaryWriter(log_dir="outputs/tb_logs/train_test_001")

        i = 0
        loss_curve = []
        if display:
            plt.ion()
            figure = plt.figure()
            plt.title("Loss")
            (plot_curve,) = plt.semilogy(np.array(loss_curve))

            gui = ti.GUI(
                "Stable Diffusion - train test",
                res=(hyperparameters["res"][0], hyperparameters["res"][1]),
            )

        # 3. Training Loop
        for epoch in range(hyperparameters["epochs"]):
            epoch_loss = 0
            for batch_idx, (images, text) in enumerate(
                tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{hyperparameters['epochs']}"
                )
            ):
                img_tmp = images[0].detach().cpu().numpy()

                images = images.to(self.device)
                context = self._do_cfg_and_tokenize_text(
                    clip, text, batch_size=hyperparameters["batch_size"]
                )
                # # Initialize random noise as latents
                latents_shape = (
                    images.size(0),
                    4,
                    LATENTS_HEIGHT,
                    LATENTS_WIDTH,
                )  # Adjust batch size and latent shape
                encoder_noise = torch.randn(
                    latents_shape, generator=generator, device=self.device
                )

                # Sample random timesteps (this would be part of the diffusion process)
                timesteps = torch.randint(
                    0,
                    hyperparameters["diffusion_steps"],
                    (images.size(0),),
                    device=self.device,
                )  # Random timesteps for each image
                # time_embeddings = get_time_embedding(timesteps).to(self.device)
                time_embeddings = get_batch_time_embedding(timesteps, self.device)

                # Add noise to the latents (simulating the diffusion process)
                with torch.no_grad():
                    latent = encoder(images, encoder_noise)
                noisy_latents = sampler.add_noise(latent, timesteps)

                # Forward pass through the diffusion model
                model_output = diffusion(noisy_latents, context, time_embeddings)

                # Compute loss (MSE between predicted noise and actual noise)
                loss = loss_fn(model_output, noisy_latents)

                # Backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # output status
                if batch_idx % 10 == 0:
                    tb_writer.add_scalar("loss", loss, batch_idx)

                if display and batch_idx % 10 == 0:
                    loss_curve.append(loss.detach().cpu().numpy())
                    plot_curve.set_xdata(np.arange(len(loss_curve)))
                    plot_curve.set_ydata(np.array(loss_curve))

                    ax = plt.gca()
                    ax.relim()
                    ax.autoscale_view()

                    figure.canvas.draw()
                    figure.canvas.flush_events()

                if batch_idx % 10 == 0 and display:
                    # Update latents by removing predicted noise
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                    latents_rmn = sampler.step(
                        timesteps[0], noisy_latents[0], model_output[0]
                    )
                    latents_rmn = latents_rmn.unsqueeze(0)
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
                    decoder.to(self.device)
                    images = decoder(latents_rmn)
                    decoder.to(
                        self.idle_device
                    )  # Free up memory by moving decoder to idle device
                    # Post-process the generated images
                    images = rescale(images, (-1, 1), (0, 255), clamp_values=True)
                    # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
                    images = images.permute(
                        0, 2, 3, 1
                    )  # Rearrange axes for image format
                    images = images.to("cpu", torch.uint8).numpy()
                    print(f"Context: {text[0]}")
                    gui.set_image(images[0])
                    gui.show()

            # Print the loss for each epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(
                f"Epoch {epoch + 1}/{hyperparameters['epochs']}, Loss: {avg_epoch_loss:.4f}"
            )

            # Optionally save the model checkpoint after each epoch
            torch.save(
                diffusion.state_dict(), f"./model_checkpoint_epoch_{epoch + 1}.pth"
            )

        print("Training complete.")


def rescale(input_tensor, source_range, target_range, clamp_values=False):
    """
    Rescales a tensor from one range to another.

    Args:
        input_tensor (torch.Tensor): The input tensor to be rescaled.
        source_range (tuple): A tuple of two values (source_min, source_max), representing
                              the range of the input tensor.
        target_range (tuple): A tuple of two values (target_min, target_max), representing
                              the desired range of the output tensor.
        clamp_values (bool, optional): If True, clamps the output tensor to the target range.
                                       Default is False.

    Returns:
        torch.Tensor: The rescaled tensor.

    Example:
        >>> x = torch.tensor([0.5, 1.0, 1.5])
        >>> rescaled_x = rescale(x, old_range=(0, 2), new_range=(-1, 1))
        >>> print(rescaled_x)
        tensor([-0.5,  0.0,  0.5])
    """
    source_min, source_max = source_range
    target_min, target_max = target_range

    # Shift the input tensor by subtracting the minimum of the source range
    input_tensor -= source_min

    # Scale the tensor to the target range
    input_tensor *= (target_max - target_min) / (source_max - source_min)

    # Adjust the tensor to the minimum of the target range
    input_tensor += target_min

    # Clamp the tensor values to the target range if required
    if clamp_values:
        input_tensor = input_tensor.clamp(target_min, target_max)

    return input_tensor
