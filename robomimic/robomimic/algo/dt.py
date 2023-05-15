import torch
import numpy as np
from torch import nn
from copy import deepcopy
import random
import torch
from torch.nn import functional as F

from robomimic.models.obs_nets import ObservationGroupEncoder, ObservationDecoder
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from collections import OrderedDict
import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.loss_utils as LossUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

#  from robomimic.algo.decision_transformer import DecisionTransformer


@register_algo_factory_func("dt")
def algo_config_to_class(algo_config):
    return DT, {}


import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size + 1, config.block_size + 1)).view(
                1, 1, config.block_size + 1, config.block_size + 1
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = "naive" if config.dt_bc_mode else "reward_conditioned"
        print(f"Using {self.model_type}")

        # input embedding stem
        #  self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size + 1, config.n_embd)
        )
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, config.max_timestep + 1, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.act_dim, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self.state_encoder = nn.Linear(config.state_dim, config.n_embd)
        #                           nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
        #  nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #  nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #                                   nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(
            nn.Linear(
                config.act_dim, config.n_embd
            ),  # nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
            nn.Tanh(),
        )
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        #  nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        #  timesteps = timesteps[:, 0,:].unsqueeze(1)
        #  print(timesteps)

        state_embeddings = self.state_encoder(
            states.type(torch.float32)
        )  # .contiguous()) # (batch * block_size, n_embd)
        #  state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)

        if actions is not None and self.model_type == "reward_conditioned":
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(
                actions.type(torch.float).squeeze(-1)
            )  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    states.shape[1] * 3 - int(targets is None),
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[
                :, -states.shape[1] + int(targets is None) :, :
            ]
        elif (
            actions is None and self.model_type == "reward_conditioned"
        ):  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == "naive":
            action_embeddings = self.action_embeddings(
                actions.type(torch.float).squeeze(-1)
            )  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    states.shape[1] * 2 - int(targets is None),
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[
                :, -states.shape[1] + int(targets is None) :, :
            ]
        elif (
            actions is None and self.model_type == "naive"
        ):  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(
            self.global_pos_emb, batch_size, dim=0
        )  # batch_size, traj_length, n_embd

        position_embeddings = (
            torch.gather(
                all_global_pos_emb,
                1,
                torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1),
            )
            + self.pos_emb[:, : token_embeddings.shape[1], :]
        )

        #          import code
        #          code.interact(local=locals())

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == "reward_conditioned":
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == "reward_conditioned":
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == "naive":
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == "naive":
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        #          loss = None
        #  if targets is not None:

        #              loss = nn.MSELoss(logits, targets)

        return logits
        #  return logits, loss


class Config:
    def __init__(self, global_config, state_dim, act_dim):
        self.dt_bc_mode = global_config.algo.dt_bc_mode
        self.block_size = global_config.train.seq_length * 3
        self.n_embd = global_config.algo.n_embed
        self.embd_pdrop = 0.1
        self.n_layer = global_config.algo.n_layers
        self.max_timestep = 1024 * 3
        self.n_head = global_config.algo.n_heads
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.state_dim = state_dim
        self.act_dim = act_dim


class DT(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()

        self.observation_group_shapes = OrderedDict()
        self.observation_group_shapes["obs"] = OrderedDict(self.obs_key_shapes)

        state_dim = sum(
            np.array(list(self.observation_group_shapes["obs"].values())).flatten()
        )
        self.config = Config(self.global_config, state_dim, self.ac_dim)

        self.nets["policy"] = GPT(self.config)
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=self.observation_group_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
        )

        #   PolicyNets.ActorNetwork(
        #  obs_shapes=self.obs_shapes,
        #  goal_shapes=self.goal_shapes,
        #  ac_dim=self.ac_dim,
        #  mlp_layer_dims=self.algo_config.actor_layer_dims,
        #  encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        #          )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        #  input_batch = dict()

        #  input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        #  input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        #  input_batch["actions"] = batch["actions"][:, 0, :].view(-1, self.config.block_size, self.config.act_dim)
        #  input_batch["returns_to_go"] = batch["returns_to_go"]
        # TODO: check shapes
        #  input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        #  input_batch["timesteps"] = batch["timesteps"].to(self.device)
        batch = TensorUtils.to_device(batch, self.device)
        #  batch["timesteps"] = batch["timesteps"].to(self.device)

        #          for k, v in batch.items():
        #  if isinstance(v, dict):
        #  for k1, v1 in v.items():
        #  print(k1, v1.shape)
        #  else:
        #  print(k, v.shape)

        #          exit()

        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DT, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        #  actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        #  print(batch.keys())
        seq_length = self.config.block_size // 3

        states = self.nets["encoder"](**batch).view(
            -1, seq_length, self.config.state_dim
        )
        actions = self.nets["policy"](
            states=states,
            actions=batch["actions"],
            targets=batch["actions"],
            rtgs=batch["returns_to_go"],
            timesteps=batch["timesteps"],
        )

        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DT, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(
    model,
    x,
    steps,
    temperature=1.0,
    sample=False,
    top_k=None,
    actions=None,
    rtgs=None,
    timesteps=None,
):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = (
            x if x.size(1) <= block_size // 3 else x[:, -block_size // 3 :]
        )  # crop context if needed
        if actions is not None:
            actions = (
                actions
                if actions.size(1) <= block_size // 3
                else actions[:, -block_size // 3 :]
            )  # crop context if needed
        rtgs = (
            rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3 :]
        )  # crop context if needed
        logits = model(
            x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps
        )
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :]  # / temperature
        # optionally crop probabilities to only the top k options
        #  if top_k is not None:
        #  logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        #  probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        #  if sample:
        #  ix = torch.multinomial(probs, num_samples=1)
        #  else:
        #  _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        #  x = ix

    return logits
