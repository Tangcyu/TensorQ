# utils/se3_diffusion.py
import torch
from typing import Tuple

# --- Centering Functions ---
def center_coords(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Centers coordinates by subtracting the center of mass (mean).
        Handles batches. Returns centered coords and the mean per batch item.
    Args:
        coords (torch.Tensor): Shape (B, N, 3) or (N, 3).
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: centered_coords, mean
            centered_coords: Shape matches input.
            mean: Shape (B, 1, 3) or (1, 3).
    """
    mean = torch.mean(coords, dim=-2, keepdim=True) # Calculate mean over N dimension
    centered_coords = coords - mean
    return centered_coords, mean

def uncenter_coords(centered_coords: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """ Adds back the center of mass. Ensures center is broadcastable."""
    # center should have shape like (B, 1, 3) or (1, 1, 3) to broadcast over (B, N, 3)
    # If center has shape (B, 3), unsqueeze dim 1.
    if center.ndim == 2 and centered_coords.ndim == 3: # center=(B,3), centered=(B,N,3)
         center = center.unsqueeze(1) # -> (B, 1, 3)
    # If center has shape (3,), unsqueeze dims 0 and 1.
    elif center.ndim == 1 and centered_coords.ndim >= 2: # center=(3,), centered=(B,N,3) or (N,3)
         # Make it (1, 1, 3) for broadcasting with (B, N, 3) or (N, 3)
         num_leading_dims = centered_coords.ndim - 1
         center_shape = (1,) * num_leading_dims + (center.shape[0],) # e.g., (1, 1, 3) or (1, 3)
         center = center.view(center_shape)

    return centered_coords + center


class SE3Diffusion:
    """ Implements the variance schedule and noise/sampling steps for SE(3) diffusion. """
    def __init__(self, timesteps: int = 1000, beta_schedule: str = 'cosine', device: str = 'cuda'):
        self.timesteps = timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu') # Safer device handling

        if beta_schedule == 'cosine':
            # Increased precision during calculation
            self.betas = self.cosine_beta_schedule(timesteps, s=0.008).to(self.device, dtype=torch.float32)
        elif beta_schedule == 'linear':
            self.betas = self.linear_beta_schedule(timesteps).to(self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        print(f"Initialized SE3Diffusion with {timesteps} steps, schedule '{beta_schedule}', device '{self.device}'")
        self._precompute_terms()

    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """ Cosine variance schedule from Ho et al. (2020) - improved version """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) # Use float64 for precision
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clamp betas to prevent numerical instability near t=0 and t=T
        return torch.clamp(betas, min=1e-6, max=0.02) # Common max beta is 0.02

    def linear_beta_schedule(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        """ Linear variance schedule """
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def _precompute_terms(self):
        """ Precompute values needed for diffusion steps. """
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])

        # For q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For q(x_{t-1} | x_t, x_0) (posterior calculation in p_sample)
        # Posterior variance: beta_t * (1 - alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t)
        # Handle potential division by zero at t=0 where alpha_cumprod=1
        variance_denom = (1.0 - self.alphas_cumprod)
        # Use where to avoid division by zero, variance is 0 at t=0 (alpha_cumprod_prev is 1)
        self.posterior_variance = torch.where(
            variance_denom > 1e-10, # Check denominator
             self.betas * (1.0 - self.alphas_cumprod_prev) / variance_denom,
             torch.zeros_like(self.betas) # Variance is zero if denominator is zero (t=0)
        )
        # Clip variance for log calculation and sampling stability
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        # Posterior mean coefficients:
        # coef1 = beta_t * sqrt(alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t)
        self.posterior_mean_coef1 = torch.where(
            variance_denom > 1e-10,
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / variance_denom,
            torch.zeros_like(self.betas) # Coef1 is zero at t=0
        )
        # coef2 = (1 - alpha_cumprod_{t-1}) * sqrt(alpha_t) / (1 - alpha_cumprod_t)
        self.posterior_mean_coef2 = torch.where(
            variance_denom > 1e-10,
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / variance_denom,
            torch.ones_like(self.betas) # Coef2 is one at t=0 (mean is just x_t)
        )

        # For calculating x_0 from x_t and noise (used if model predicts noise)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)


    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """ Extracts the value 'a' at indices 't' and reshapes it for broadcasting. """
        batch_size = t.shape[0]
        # Ensure t is on the same device as 'a' and Long type
        out = a.gather(-1, t.to(a.device).long())
        # Reshape to (B, 1, 1, ...) for broadcasting, matching x_shape's dimensions
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adds noise to the input coordinates according to the schedule q(x_t | x_0).
        IMPORTANT: Centers the coordinates *before* adding noise.

        Args:
            x_start (torch.Tensor): The initial clean data (e.g., coordinates), shape (B, N, 3).
                                    Can be uncentered.
            t (torch.Tensor): Timesteps for each batch element, shape (B,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_noisy_centered: Noisy coordinates (CENTERED). Shape matches input.
                - noise: The sampled Gaussian noise added (relative to centered data). Shape matches input.
                - center: The original center of mass for each structure. Shape (B, 1, 3).
        """
        x_centered, center = center_coords(x_start) # Center FIRST
        noise = torch.randn_like(x_centered, device=self.device) # Noise matches centered shape

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_centered.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_centered.shape)

        # Apply noise in the centered space: x_t = sqrt(alpha_c)*x_0_centered + sqrt(1-alpha_c)*noise
        x_noisy_centered = sqrt_alphas_cumprod_t * x_centered + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy_centered, noise, center # Return CENTERED noisy version and the center


    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """ Calculates the predicted x_0 given x_t (assumed centered) and the predicted noise (eps). """
        # Ensure x_t is treated as centered
        assert x_t.ndim == 3 and x_t.shape[-1] == 3

        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        # x_0_pred = (x_t - sqrt(1-alpha_c)*eps_pred) / sqrt(alpha_c)
        x0_pred_centered = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred
        return x0_pred_centered # Predicted x0 will also be centered


    def p_sample(self, model_output_x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Performs one step of the reverse diffusion process q(x_{t-1} | x_t, x_0_pred).
        Assumes model_output_x0 is the predicted x_0 (CENTERED).
        Assumes x_t is the current noisy coordinates (CENTERED).

        Args:
            model_output_x0 (torch.Tensor): Predicted x_0 (CENTERED) from the model, shape (B, N, 3).
            x_t (torch.Tensor): Current noisy coordinates (CENTERED), shape (B, N, 3).
            t (torch.Tensor): Current timestep for each batch element, shape (B,).
            noise_scale (float): Scale factor for the sampling noise (0 for deterministic DDIM-like).

        Returns:
            torch.Tensor: Sampled coordinates for the previous timestep x_{t-1} (CENTERED).
        """
        # Ensure inputs are treated as centered
        assert model_output_x0.ndim == 3 and model_output_x0.shape[-1] == 3
        assert x_t.ndim == 3 and x_t.shape[-1] == 3
        x0_pred_centered = model_output_x0 # Input from model should already be centered x0 prediction

        # Get posterior parameters for current timestep t
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # Calculate posterior mean: E[x_{t-1} | x_t, x_0_pred]
        # posterior_mean = coef1 * x_0_pred + coef2 * x_t
        posterior_mean = posterior_mean_coef1_t * x0_pred_centered + posterior_mean_coef2_t * x_t

        # Sample noise for the step (scaled by noise_scale)
        noise = torch.randn_like(x_t, device=self.device) if noise_scale > 0 else torch.zeros_like(x_t, device=self.device)

        # Calculate x_{t-1} = posterior_mean + sqrt(posterior_variance) * noise
        # Use log variance for numerical stability: exp(0.5 * log_var) = sqrt(var)
        # Don't add noise at t=0 (variance is zero)
        nonzero_mask = (t != 0).float().view(-1, *((1,) * (len(x_t.shape) - 1))) # Mask where t > 0
        # Ensure mask is on the correct device
        nonzero_mask = nonzero_mask.to(self.device)

        x_prev_centered = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance_t) * noise * noise_scale

        return x_prev_centered # Return the CENTERED previous step