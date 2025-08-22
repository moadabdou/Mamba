"""

This script provides a basic and simplified implementation of the Mamba architecture,
a modern state-space model (SSM) for efficient and effective sequence modeling.

The implementation includes four main components:
1.  Embedder: Handles token embedding and projection back to vocabulary logits.
2.  SSM (State Space Model): The core selective state-space layer.
3.  MambaBlock: A single block combining the SSM with normalization, convolutions, and gating.
4.  BasicMambaModel: The final model that stacks multiple MambaBlocks.

This code is intended for educational purposes to illustrate the key concepts of Mamba.
"""

import torch
from torch import nn
from torch.nn import functional as F

# ======================================================================================
# 1. Embedding Layer
# ======================================================================================

class Embedder(nn.Module):
    """
    Handles the embedding of input token indices and the projection of the model's
    output back to vocabulary logits.
    """
    def __init__(self, vocab_size: int, emb_dim: int):
        """
        Initializes the Embedder module.

        Args:
            vocab_size (int): The size of the vocabulary.
            emb_dim (int): The dimensionality of the token embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # Embedding layer: maps token indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        # "De-embedding" layer: maps hidden states back to vocabulary logits
        self.dembeddings = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Note: The primary logic is split into encode() and decode() methods for clarity.
        This forward method is not used directly in the BasicMambaModel.
        """
        pass

    def encode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of token indices into dense embedding vectors.

        Args:
            indices (torch.Tensor): A tensor of shape (B, L) containing token indices.

        Returns:
            torch.Tensor: The corresponding embeddings of shape (B, L, D).
        """
        return self.embeddings(indices)

    def decode(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden state embeddings back into vocabulary logits.

        Args:
            emb (torch.Tensor): A tensor of shape (B, L, D) containing hidden states.

        Returns:
            torch.Tensor: The corresponding logits of shape (B, L, vocab_size).
        """
        return self.dembeddings(emb)

# ======================================================================================
# 2. Core State Space Model (SSM)
# ======================================================================================

class SSM(nn.Module):
    """
    Implements a simplified selective State Space Model (SSM) layer.

    This layer models a sequence by updating a hidden state 'h' based on the input 'x'
    using a discretized version of a continuous-time system defined by (A, B, C) matrices.
    The key feature is that the B, C, and discretization step-size (delta) are
    dynamically generated from the input, making the model "selective".
    """
    def __init__(self, hidden: int, emb_dim: int):
        """
        Initializes the SSM module.

        Args:
            hidden (int): The size of the per-dimension hidden state (often denoted as N).
            emb_dim (int): The dimensionality of the input/output embeddings (D).
        """
        super().__init__()
        self.hidden = hidden
        self.emb_dim = emb_dim

        # State matrix (A), initialized to be diagonal and stable (negative).
        # This is a core parameter of the continuous-time system.
        self.A = nn.Parameter(-torch.exp(torch.randn(emb_dim, hidden)))

        # Projection layers to derive input-dependent B, C, and delta.
        self.B_layer = nn.Linear(emb_dim, hidden)
        self.C_layer = nn.Linear(emb_dim, hidden)
        self.delta_layer = nn.Linear(emb_dim, 1)

        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the SSM.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D), where B is batch size,
                              L is sequence length, and D is embedding dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape (B, L, D).
        """
        Bsz, L, D = x.shape
        device = x.device
        assert D == self.emb_dim, "Input embedding dim must match emb_dim"

        # --- 1. Calculate Input-Dependent Parameters ---
        # Project input 'x' to get dynamic B, C, and delta for each time step.
        B_in = self.B_layer(x)      # (B, L, H)
        C_in = self.C_layer(x)      # (B, L, H)
        delta = self.softplus(self.delta_layer(x))  # (B, L, 1), ensures delta > 0

        # --- 2. Reshape for Per-Dimension SSM ---
        # The SSM operates independently on each of the D embedding dimensions.
        B_in = B_in.unsqueeze(2).expand(Bsz, L, D, self.hidden)
        C_in = C_in.unsqueeze(2).expand(Bsz, L, D, self.hidden)
        delta = delta.unsqueeze(2).expand(Bsz, L, D, self.hidden)

        # --- 3. Discretization (from continuous A, B to discrete Ad, Bd) ---
        # This uses the Zero-Order Hold (ZOH) method.
        # Ad = exp(delta * A)
        A = self.A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H)
        Ad = torch.exp(delta * A)             # (B, L, D, H)

        # Bd is calculated using a stable approximation of (Ad - 1)/A * B
        eps = 1e-6
        denom = -A
        frac = torch.where(
            denom.abs() > eps,
            (1.0 - Ad) / denom,
            delta * (1 - 0.5 * delta * A + (A * delta)**2 / 6.0) # Taylor approx for small denom
        )
        Bd = frac * B_in  # (B, L, D, H)

        # --- 4. Recurrence Calculation (Sequential Scan) ---
        # This loop iterates through the sequence length to update the hidden state.
        # h_t = Ad_t * h_{t-1} + Bd_t * x_t
        h = torch.zeros(Bsz, D, self.hidden, device=device)
        outputs = []
        for t in range(L):
            x_t = x[:, t, :].unsqueeze(-1)  # (B, D, 1)
            h = Ad[:, t] * h + Bd[:, t] * x_t
            # y_t = C_t * h_t
            y_t = (h * C_in[:, t]).sum(-1)  # (B, D)
            outputs.append(y_t)

        # Stack the outputs from each time step
        y = torch.stack(outputs, dim=1)  # (B, L, D)
        return y

# ======================================================================================
# 3. Mamba Block
# ======================================================================================

class MambaBlock(nn.Module):
    """
    A single Mamba block, which is the main building block of the Mamba model.

    This block integrates the SSM with modern neural network components:
    - Pre-normalization (LayerNorm)
    - Gated activation (SiLU)
    - 1D convolution for local context
    - A residual connection for stable training
    """
    def __init__(self, up_proj_dim: int, state_space_dim: int, emb_dim: int):
        """
        Initializes the MambaBlock.

        Args:
            up_proj_dim (int): The expanded dimensionality inside the block.
            state_space_dim (int): The hidden state size (N) for the SSM.
            emb_dim (int): The input and output embedding dimension (D).
        """
        super().__init__()
        self.up_proj_dim = up_proj_dim
        self.state_space_dim = state_space_dim
        self.emb_dim = emb_dim
        self.conv1d_kernel_size = 4

        # --- Layers ---
        self.norm = nn.LayerNorm(emb_dim)
        # Projects input from D to 2 * up_proj_dim for gating
        self.in_proj = nn.Linear(emb_dim, 2 * up_proj_dim)
        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=up_proj_dim,
            out_channels=up_proj_dim,
            kernel_size=self.conv1d_kernel_size,
            padding=0, # Handled manually with F.pad
            groups=up_proj_dim # Depthwise convolution
        )
        # The core SSM layer
        self.ssm = SSM(state_space_dim, up_proj_dim)
        # Projects output back from up_proj_dim to D
        self.out_proj = nn.Linear(up_proj_dim, emb_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MambaBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, L, D).
        """
        # Store original input for the final residual connection
        residual = x

        # 1. Pre-Normalization
        x = self.norm(x)

        # 2. Input Projection and Gating Split
        # Project x and split it into two paths: one for the SSM and one for the gate.
        x_proj = self.in_proj(x)
        x_ssm, x_gate = x_proj.split(self.up_proj_dim, dim=-1)

        # Activate the gating branch
        x_gate_act = self.activation(x_gate)

        # 3. SSM Path
        # The main data path involves a convolution followed by the SSM.
        x_ssm_t = x_ssm.transpose(-2, -1)  # (B, L, D) -> (B, D, L) for Conv1d
        # Pad for causal convolution
        x_padded = F.pad(x_ssm_t, (self.conv1d_kernel_size - 1, 0))
        x_conv = self.conv1d(x_padded)
        x_conv_t = x_conv.transpose(-2, -1) # (B, D, L) -> (B, L, D)
        x_conv_act = self.activation(x_conv_t)
        x_ssm_out = self.ssm(x_conv_act)

        # 4. Combine Paths and Final Projection
        # Modulate the SSM output with the activated gate (element-wise multiplication)
        x_combined = x_ssm_out * x_gate_act
        # Project back to the original embedding dimension
        x_out = self.out_proj(x_combined)

        # 5. Add Residual Connection
        return x_out + residual

# ======================================================================================
# 4. Top-Level Mamba Model
# ======================================================================================

class BasicMambaModel(nn.Module):
    """
    A basic Mamba model composed of an embedding layer, a stack of MambaBlocks,
    and a final projection layer to produce logits.
    """
    def __init__(self, n_blocks: int, up_proj_dim: int, state_space_dim: int, emb_dim: int, vocab_size: int):
        """
        Initializes the BasicMambaModel.

        Args:
            n_blocks (int): The number of MambaBlocks to stack.
            up_proj_dim (int): The expanded dimensionality inside each MambaBlock.
            state_space_dim (int): The hidden state size (N) for the SSM in each block.
            emb_dim (int): The model's primary embedding dimension (D).
            vocab_size (int): The size of the vocabulary for the embedder.
        """
        super().__init__()
        self.up_proj_dim = up_proj_dim
        self.state_space_dim = state_space_dim
        self.emb_dim = emb_dim
        self.n_blocks = n_blocks

        # --- Layers ---
        self.embedder = Embedder(vocab_size, emb_dim)
        self.blocks = nn.ModuleList(
            [MambaBlock(up_proj_dim, state_space_dim, emb_dim) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass for the model.

        Args:
            indices (torch.Tensor): Input tensor of token indices, shape (B, L).

        Returns:
            torch.Tensor: Output logits tensor of shape (B, L, vocab_size).
        """
        # 1. Get token embeddings
        x = self.embedder.encode(indices)

        # 2. Pass through all MambaBlocks
        for block in self.blocks:
            x = block(x)

        # 3. Final normalization
        x = self.out_norm(x)

        # 4. Decode to vocabulary logits
        logits = self.embedder.decode(x)

        return logits