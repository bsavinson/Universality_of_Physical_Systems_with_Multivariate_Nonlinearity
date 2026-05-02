import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_symmetric(A):
    """Project a real matrix, with optional batch dimensions, to symmetric."""
    return 0.5 * (A + A.transpose(-1, -2))


def forward_with_params(x, a, b, S, I_in, mask, amplitude_encoding=False):
    """
    Forward pass for the MNIST/FashionMNIST free-space model.

    The model phase-encodes a 14x14 input, solves the free-space linear system,
    applies a Fourier lens, and reads out the central detector pixel.
    """
    batch_size, input_dim = x.shape
    nsamp, replications, dim = a.shape[0], a.shape[2], a.shape[-1]
    n = mask.shape[-1]

    input_dim_axis = int(input_dim**0.5)
    if input_dim_axis**2 != input_dim:
        raise ValueError("Input dimension must be square.")
    if dim % input_dim != 0:
        raise ValueError("Model dimension must be divisible by input dimension.")

    reps = dim // input_dim
    replications_per_dim = int(reps**0.5)
    if replications_per_dim**2 != reps:
        raise ValueError("Number of input replications must be square.")

    x = x.reshape(batch_size, input_dim_axis, input_dim_axis)
    x = x.repeat([1, replications_per_dim, replications_per_dim])
    x = x.reshape(batch_size, dim)
    x = x.unsqueeze(1).repeat([1, replications, 1])
    x = x.unsqueeze(0)

    if amplitude_encoding:
        T_diag = torch.sigmoid(a * x + b)
    else:
        T_diag = torch.exp(1j * (x * a + b))

    Id = torch.eye(dim, dtype=torch.cfloat, device=x.device).reshape(1, 1, 1, dim, dim)
    TST = torch.einsum("nbri,nrij,nbrj->nbrij", T_diag, S, T_diag)
    M = 2 * (Id - 0.5 * TST)

    I_in = I_in.unsqueeze(1).repeat([1, batch_size, 1, 1])
    batch_shape = M.shape[:-2]
    I_out = torch.linalg.solve(M.reshape(-1, dim, dim), I_in.reshape(-1, dim))
    I_out = I_out.reshape(*batch_shape[:-1], n, n)

    I_out_fft = torch.fft.fft2(I_out.unsqueeze(2), dim=(-2, -1), norm="ortho")
    I_out_fft = torch.fft.fftshift(I_out_fft, dim=(-2, -1))
    I_out_power = torch.abs(I_out_fft) ** 2

    masked_outputs = I_out_power * mask.unsqueeze(0).unsqueeze(0)
    return masked_outputs.sum(dim=(-2, -1))


class FreeSpaceModel(nn.Module):
    """
    Supported configuration:
    - 14x14 MNIST or FashionMNIST inputs
    - one optical layer
    - ten independent multi-lens readout channels
    - trainable or random S matrices
    - phase encoding by default
    """

    def __init__(
        self,
        model_n,
        train_S=True,
        replications_per_dim=1,
        amplitude_encoding=False,
    ):
        super().__init__()

        if model_n < 14:
            raise ValueError("model_n must be at least 14 for MNIST/FashionMNIST.")

        self.model_n = model_n
        self.dim = model_n**2
        self.replications = int(replications_per_dim**2)
        self.n = replications_per_dim * model_n
        self.train_S = train_S
        self.amplitude_encoding = amplitude_encoding

        a = torch.zeros(10, 1, self.replications, self.dim, dtype=torch.float, device=device)
        for r in range(self.replications):
            a[:, :, r, :] = torch.randn(1, self.dim, dtype=torch.float, device=device) * 0.1 * 10**r
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(
            torch.randn(10, 1, self.replications, self.dim, dtype=torch.float, device=device)
        )
        self.I_in = nn.Parameter(
            torch.randn(10, self.replications, self.dim, dtype=torch.cfloat, device=device)
        )

        S_param = to_symmetric(
            torch.randn(
                10,
                self.replications,
                self.dim,
                self.dim,
                dtype=torch.float,
                device=device,
            )
        )
        if train_S:
            self.S_param = nn.Parameter(S_param)
        else:
            self.register_buffer("S_param", S_param)
            self.register_buffer("_S", torch.matrix_exp(1j * self.S_param))

        self.mask = self.build_center_mask().to(device)
        self._eval_S_cache = None

    def S(self, S_param=None):
        if S_param is None and not self.train_S:
            return self._S
        if S_param is None:
            S_param = self.S_param
        return torch.matrix_exp(1j * to_symmetric(S_param))

    def train(self, mode=True):
        self._eval_S_cache = None
        return super().train(mode)

    def class_S(self, class_index):
        if not self.training and self.train_S:
            if self._eval_S_cache is None:
                self._eval_S_cache = self.S().detach()
            return self._eval_S_cache[class_index]
        if not self.train_S:
            return self._S[class_index]
        return self.S(self.S_param[class_index])

    def build_center_mask(self):
        mask = torch.zeros((1, self.n, self.n), dtype=torch.float)
        pixel_location = int(self.n / 2)
        mask[:, pixel_location, pixel_location] = 1.0
        return mask

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.zeros((batch_size, 10), dtype=torch.float, device=x.device)

        for i in range(10):
            output[:, i] = forward_with_params(
                x,
                self.a[i].unsqueeze(0),
                self.b[i].unsqueeze(0),
                self.class_S(i).unsqueeze(0),
                self.I_in[i].unsqueeze(0),
                self.mask.to(x.device),
                self.amplitude_encoding,
            ).squeeze(0).squeeze(-1)

        return output
