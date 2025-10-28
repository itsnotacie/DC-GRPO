import torch
import torch.nn.functional as F
from trl import GRPOTrainer
from typing import Optional, Dict, Any

def compute_dynamic_clip_bounds(advantages, probabilities, base_epsilon=0.2, lambda_max=0.15, gamma=1.8):
    """
    Compute dynamic clipping boundaries
    
    Args:
        advantages: Advantage values (batch_size,)
        probabilities: Generation probabilities (batch_size,)
        base_epsilon: Base clipping coefficient
        lambda_max: Maximum clipping adjustment magnitude
        gamma: Power exponent for probability adjustment
    
    Returns:
        epsilon_low, epsilon_high: Dynamic clipping boundaries
    """
    # Step 1: Adjust probabilities based on advantage values
    pi = torch.where(
        advantages <= 0,
        probabilities,      # Low reward: use original probability
        1 - probabilities   # High reward: use 1 - probability
    )
    
    # Step 2: Calculate clipping weight δ_i
    # Left term: sigmoid mapping of absolute advantage value
    advantage_weight = 1 / (1 + torch.exp(-torch.abs(advantages)))
    
    # Right term: power of probability
    prob_weight = torch.pow(pi, gamma)
    
    # Combined weight
    delta = advantage_weight * prob_weight
    
    # Step 3: Calculate dynamic clipping coefficients
    # Sign function
    sign_advantage = torch.sign(advantages)
    
    # Calculate epsilon_low and epsilon_high
    epsilon_low = base_epsilon + lambda_max * delta * (-sign_advantage)
    epsilon_high = base_epsilon + lambda_max * delta * sign_advantage
    
    return epsilon_low, epsilon_high


def compute_dc_grpo_loss(
    old_logprobs,
    new_logprobs,
    advantages,
    old_probs,  # Need to pass in old policy probabilities
    base_epsilon=0.2,
    lambda_max=0.15,
    gamma=1.8
):
    """
    Compute DC-GRPO loss
    
    Args:
        old_logprobs: Log probabilities of old policy
        new_logprobs: Log probabilities of new policy
        advantages: Advantage values
        old_probs: Probabilities of old policy (for dynamic clipping)
        base_epsilon: Base clipping coefficient
        lambda_max: Maximum clipping adjustment magnitude
        gamma: Power exponent for probability adjustment
    """
    # Calculate probability ratio
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # Compute dynamic clipping boundaries
    epsilon_low, epsilon_high = compute_dynamic_clip_bounds(
        advantages, 
        old_probs, 
        base_epsilon, 
        lambda_max, 
        gamma
    )
    
    # Dynamic clipping
    ratio_clipped = torch.clamp(
        ratio,
        1 - epsilon_low,
        1 + epsilon_high
    )
    
    # Calculate loss
    loss_unclipped = ratio * advantages
    loss_clipped = ratio_clipped * advantages
    
    # Take minimum value
    loss = torch.min(loss_unclipped, loss_clipped)
    
    return -loss.mean() 


class DCGRPOTrainer(GRPOTrainer):
    """
    Dynamic Clip GRPO Trainer
    """
    def __init__(
        self,
        *args,
        lambda_max: float = 0.15,
        gamma: float = 1.8,
        base_epsilon: float = 0.2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_max = lambda_max
        self.gamma = gamma
        self.base_epsilon = base_epsilon
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ):
        """
        Override loss computation function to use dynamic clipping
        """
        # Get old policy outputs
        with torch.no_grad():
            old_outputs = self.ref_model(**inputs)
            old_logprobs = old_outputs.logits.log_softmax(dim=-1)
            old_probs = old_outputs.logits.softmax(dim=-1)
        
        # Get new policy outputs
        new_outputs = model(**inputs)
        new_logprobs = new_outputs.logits.log_softmax(dim=-1)
        
        # Get advantage values (assumed to be pre-computed in inputs)
        advantages = inputs.get("advantages")
        
        # Extract probabilities for corresponding tokens
        # This needs to extract based on actual token indices
        input_ids = inputs["input_ids"]
        
        # Compute DC-GRPO loss
        loss = compute_dc_grpo_loss(
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            advantages=advantages,
            old_probs=old_probs,
            base_epsilon=self.base_epsilon,
            lambda_max=self.lambda_max,
            gamma=self.gamma
        )
        
        return (loss, new_outputs) if return_outputs else loss
