# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F
from tqdm import tqdm

from flow_matching.path import MixtureDiscreteProbPath, MixtureDiscreteSoftmaxProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from .utils import get_nearest_times


class MixtureDiscreteEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Sample a sequence of discrete values from the given model.

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import MixtureDiscreteEulerSolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return ...

            model = DummyModel()
            solver = MixtureDiscreteEulerSolver(model=model)

            x_init = torch.LongTensor([122, 725])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): The initial state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. If None then time discretization is set to be time_grid.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): The CTMC process is solved in the interval [time_grid[0], time_grid[-1]] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        steps_counter = 0
        res = []

        if return_intermediates:
            res = [x_init.clone()]

        if verbose:
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_1 = categorical(p_1t.to(dtype=dtype_categorical))

                # Checks if final step
                if i == n_steps - 1:
                    x_t = x_1
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.path.scheduler(t=t)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(
                        k_t.dtype
                    )
                    u = d_k_t / (1 - k_t) * delta_1

                    # Add divergence-free part
                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_1
                        )

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )

                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h * intensity)

                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            u[mask_jump].to(dtype=dtype_categorical)
                        )

                steps_counter += 1
                t = t + h

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        if return_intermediates:
            if step_size is None:
                return torch.stack(res, dim=0)
            else:
                return torch.stack(res, dim=0)[order]
        else:
            return x_t


class MixtureDiscreteSoftmaxEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path_txt: MixtureDiscreteSoftmaxProbPath,
        path_img: MixtureDiscreteSoftmaxProbPath,
        vocabulary_size_txt: int,
        vocabulary_size_img: int,
    ):
        super().__init__()
        self.model = model
        self.path_txt = path_txt
        self.path_img = path_img
        self.vocabulary_size_txt = vocabulary_size_txt
        self.vocabulary_size_img = vocabulary_size_img

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        # callback: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Sample a sequence of discrete values from the given model.

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import MixtureDiscreteEulerSolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return ...

            model = DummyModel()
            solver = MixtureDiscreteEulerSolver(model=model)

            x_init = torch.LongTensor([122, 725])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): The initial state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. If None then time discretization is set to be time_grid.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): The CTMC process is solved in the interval [time_grid[0], time_grid[-1]] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        steps_counter = 0
        res = []

        if return_intermediates:
            if self.model.g_or_u == 'generation':
                res = [x_init.clone()[model_extras['datainfo']['image_token_mask']==1].reshape(x_init.shape[0], -1)]
            elif self.model.g_or_u =='understanding':
                res = [x_init.clone()[model_extras['datainfo']['text_token_mask']==1].reshape(x_init.shape[0], -1)]
            else:
                res = [x_init.clone()]


        if verbose:
            ctx = tqdm(total=time_grid[-1].item(), desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            original_x_t = x_t.clone()
            batch_size = original_x_t.shape[0]
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                p_1t_txt, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                if p_1t_txt is None:
                    x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                    x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    # x_1 = x_1 * data_info['image_token_mask'] + x_t * (1 - data_info['image_token_mask']) 
                elif p_1t_img is None:
                    x_1 = categorical(p_1t_txt.to(dtype=dtype_categorical))
                    x_1 = x_1[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    x_t = x_t[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    # x_1 = x_1 * data_info['text_token_mask'] + x_t * (1 - data_info['text_token_mask']) 
                else:
                    x_1_img = categorical(p_1t_img.to(dtype=dtype_categorical))
                    x_1_txt = categorical(p_1t_txt.to(dtype=dtype_categorical))
                    x_1_img = x_1_img[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_1_txt = x_1_txt[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    x_t_img = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_t_txt = x_t[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    # x_1_txt = x_1_txt * data_info['text_token_mask'] + x_t * (1 - data_info['text_token_mask']) 
                    # x_1_img = x_1_img * data_info['image_token_mask'] + x_t * (1 - data_info['image_token_mask']) 
                    # x_1 = x_1_txt * (1 - data_info['image_token_mask']) + x_1_img * data_info['image_token_mask'] 


                # Checks if final step
                if i == n_steps - 1:
                    if p_1t_txt is None:
                        x_t = x_1
                    elif p_1t_img is None:
                        x_t = x_1
                    else:
                        x_t = original_x_t.clone()
                        x_t[data_info['image_token_mask']==1] = x_1_img.flatten()
                        x_t[data_info['text_token_mask']==1] = x_1_txt.flatten()

                    if return_intermediates:
                        res.append(x_t.clone())
                else:
                    if p_1t_txt is None:
                        # Compute p_t(x|x_1)
                        emb_x_1 = self.path_img.embedding(x_1)
                        prob_x_t = self.path_img.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_img.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_img.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_img.c * self.path_img.a * ((t / (1 - t)) ** (self.path_img.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_img)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        # print(f"intensity:{intensity.sum()}")
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        if return_intermediates:
                            res.append(x_t.clone())
                            # if callback:
                            #     yield x_t
                            # res.append(x_1.clone())
                        original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                        # original_x_t[data_info['image_token_mask']==1] = x_1.flatten()
                        x_t = original_x_t.clone()
                    elif p_1t_img is None:
                        # Compute p_t(x|x_1)
                        emb_x_1 = self.path_txt.embedding(x_1)
                        prob_x_t = self.path_txt.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_txt.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_txt.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_txt.c * self.path_txt.a * ((t / (1 - t)) ** (self.path_txt.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_txt)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        if return_intermediates:
                            res.append(x_t.clone())
                            # if callback:
                            #     yield x_t
                        original_x_t[data_info['text_token_mask']==1] = x_t.flatten()
                        x_t = original_x_t.clone()
                    else:
                        # The text part
                        x_t = x_t_txt.clone()
                        x_1 = x_1_txt.clone()
                        # Compute p_t(x|x_1)
                        emb_x_1 = self.path_txt.embedding(x_1)
                        prob_x_t = self.path_txt.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_txt.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_txt.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_txt.c * self.path_txt.a * ((t / (1 - t)) ** (self.path_txt.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_txt)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        original_x_t[data_info['text_token_mask']==1] = x_t.flatten()

                        # The image part
                        x_t = x_t_img.clone()
                        x_1 = x_1_img.clone()
                        scheduler_output = self.path_img.scheduler(t=t)
                        emb_x_1 = self.path_img.embedding(x_1)
                        prob_x_t = self.path_img.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_img.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_img.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_img.c * self.path_img.a * ((t / (1 - t)) ** (self.path_img.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_img)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                        
                        x_t = original_x_t.clone()
                        if return_intermediates:
                            res.append(x_t.clone())

                steps_counter += 1
                t = t + h

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        # if return_intermediates and not callback:
        if return_intermediates:
            return torch.stack(res, dim=0)[:, 0, :].reshape(n_steps+1, -1)
        # elif callback:
        #     yield x_t
        else:
            return x_t
