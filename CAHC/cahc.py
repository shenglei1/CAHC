from __future__ import annotations

import time

import torch
from torch import nn
import torch.nn.functional as F

from gnn_layers import GAT, HyperGraph


class CAHC(nn.Module):


    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.device = torch.device(
            "cuda", index=args.gpu
        ) if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)
        self.memory = nn.LSTMCell(args.hid_size, args.hid_size).double()
        self.coalition_encoder = HyperGraph(args)
        self.topology_decoder = nn.Sequential(
            nn.Linear(2 * args.hid_size, args.hid_size // 2),
            nn.ReLU(),
            nn.Linear(args.hid_size // 2, args.hid_size // 8),
            nn.ReLU(),
            nn.Linear(args.hid_size // 8, 2),
        )
        self.distributor = GAT(
            args.hid_size,
            args.gat_hid_size,
            dropout=0.0,
            negative_slope=0.2,
            num_heads=args.gat_num_heads,
            self_loop_type=args.self_loop_type,
            average=False,
            normalize=True,
        )
        self.message_decoder = nn.Linear(args.hid_size, args.hid_size)
        self.action_heads = nn.ModuleList(
            [nn.Linear(2 * args.hid_size, actions) for actions in args.naction_heads]
        )
        self.value_head = nn.Linear(2 * args.hid_size, 1)
        self.last_diagnostics = {}
        self.to(self.device)

    def init_hidden(self, batch_size: int):
        shape = (batch_size * self.nagents, self.hid_size)
        return (
            torch.zeros(shape, device=self.device, dtype=torch.float64),
            torch.zeros(shape, device=self.device, dtype=torch.float64),
        )

    def _agent_mask(self, info: dict) -> torch.Tensor:
        if "alive_mask" in info:
            mask = torch.as_tensor(
                info["alive_mask"], device=self.device, dtype=torch.get_default_dtype()
            )
        else:
            mask = torch.ones(
                self.nagents, device=self.device, dtype=torch.get_default_dtype()
            )
        return mask.view(self.nagents, 1)

    def _set_capabilities(self, info: dict) -> None:
        levels = info.get("agent_level")
        if levels is None:


            return
        tensor = torch.as_tensor(levels, device=self.device, dtype=torch.long)
        self.coalition_encoder.agent_levels = tensor.view(1, self.nagents)

    def _decode_topology(
        self,
        features: torch.Tensor,
        agent_mask: torch.Tensor,
        physical_support: torch.Tensor | None = None,
        force_physical_links: bool = False,
    ) -> torch.Tensor:
        n_agents, width = features.shape
        if force_physical_links:
            if physical_support is None:
                raise ValueError("Forced physical links require registered support.")
            adjacency = torch.as_tensor(
                physical_support, device=features.device, dtype=features.dtype
            )
        elif self.args.complete_topology:
            adjacency = torch.ones(
                n_agents, n_agents, device=features.device, dtype=features.dtype
            )
        else:
            source = features.unsqueeze(1).expand(n_agents, n_agents, width)
            target = features.unsqueeze(0).expand(n_agents, n_agents, width)
            pair_features = torch.cat([source, target], dim=-1)
            logits = self.topology_decoder(pair_features)
            if not self.args.directed:
                logits = 0.5 * (logits + logits.transpose(0, 1))
            choice = F.gumbel_softmax(
                logits,
                tau=self.args.topology_temperature,
                hard=True,
                dim=-1,
            )
            adjacency = choice[..., 1]

        active = agent_mask.expand(n_agents, n_agents)
        adjacency = adjacency * active * active.transpose(0, 1)


        adjacency = adjacency * (
            1.0 - torch.eye(n_agents, device=adjacency.device, dtype=adjacency.dtype)
        )
        if physical_support is not None:
            support = torch.as_tensor(
                physical_support,
                device=adjacency.device,
                dtype=adjacency.dtype,
            )
            if tuple(support.shape) != (n_agents, n_agents):
                raise ValueError(
                    "Physical communication support must be an N-by-N matrix."
                )
            support = (support > 0).to(adjacency.dtype)
            support = support * (
                1.0
                - torch.eye(
                    n_agents, device=adjacency.device, dtype=adjacency.dtype
                )
            )
            adjacency = adjacency * support
        return adjacency

    def forward(self, inputs, info=None):
        if info is None:
            info = {}
        observations, memory_state = inputs
        start = time.perf_counter()
        encoded = self.obs_encoder(observations)
        hidden, cell = self.memory(encoded.squeeze(0), memory_state)

        self._set_capabilities(info)
        collect_diagnostics = bool(info.get("collect_model_diagnostics", False))
        self.coalition_encoder.collect_wave_diagnostics = collect_diagnostics
        self.coalition_encoder.collect_spectral_diagnostics = bool(
            collect_diagnostics and info.get("record_spectral_diagnostics", False)
        )
        fixed_incidence = info.get("probe2_fixed_incidence")
        refined, incidence = self.coalition_encoder(
            observations, fixed_incidence=fixed_incidence
        )
        agent_mask = self._agent_mask(info)
        refined = refined.squeeze(0) * agent_mask
        physical_support = None
        if fixed_incidence is not None:
            registered = incidence.squeeze(0)
            physical_support = torch.matmul(registered.t(), registered)
        adjacency = self._decode_topology(
            refined,
            agent_mask,
            physical_support=physical_support,
            force_physical_links=bool(info.get("probe2_force_physical_links", False)),
        )
        adjacency_before_dropout = adjacency
        keep_mask = info.get("evaluation_link_keep_mask")
        if keep_mask is not None:
            keep_mask = torch.as_tensor(
                keep_mask, device=adjacency.device, dtype=adjacency.dtype
            )
            if tuple(keep_mask.shape) != tuple(adjacency.shape):
                raise ValueError(
                    "Evaluation link keep mask must match the decoded adjacency."
                )
            adjacency = adjacency * keep_mask
        messages, attention = self.distributor(refined, adjacency)
        messages = self.message_decoder(F.elu(messages)) * agent_mask

        joint = torch.cat([hidden, messages], dim=-1).unsqueeze(0)
        values = self.value_head(joint).view(-1)
        action_out = [
            F.log_softmax(head(joint), dim=-1) for head in self.action_heads
        ]

        if collect_diagnostics:
            wave = self.coalition_encoder.last_wave_diagnostics
            self.last_diagnostics = {
                "incidence": incidence.detach(),
                "adjacency": adjacency.detach(),
                "adjacency_before_dropout": adjacency_before_dropout.detach(),
                "attention": attention.detach(),
                "visible_task_requirement": (
                    self.coalition_encoder.last_visible_task_requirement
                ),
                "coalition_cardinality": (
                    self.coalition_encoder.last_coalition_cardinality
                ),
                "target_partner_capability": (
                    self.coalition_encoder.last_target_partner_capability
                ),
                "wave": wave,
            }
            info["model_diagnostics"] = self.last_diagnostics
            info["inference_delay_ms"] = 1000.0 * (time.perf_counter() - start)
        else:
            self.last_diagnostics = {}
        return action_out, values, (hidden, cell)
