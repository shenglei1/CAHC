


from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GAT(nn.Module):


    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        negative_slope: float,
        num_heads: int = 1,
        bias: bool = True,
        self_loop_type: int = 2,
        average: bool = False,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.dropout = dropout
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize
        self.W = nn.Parameter(torch.empty(in_features, num_heads * out_features))
        self.a_i = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_j = nn.Parameter(torch.empty(num_heads, out_features, 1))
        output_width = out_features if average else num_heads * out_features
        self.bias = nn.Parameter(torch.empty(output_width)) if bias else None
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.a_i, gain=gain)
        nn.init.xavier_normal_(self.a_j, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor):
        hidden = torch.mm(features, self.W).view(
            -1, self.num_heads, self.out_features
        )
        n_agents = hidden.size(0)
        if self.self_loop_type == 0:
            adjacency = adjacency * (
                torch.ones_like(adjacency)
                - torch.eye(n_agents, device=adjacency.device)
            )
        elif self.self_loop_type == 1:
            adjacency = torch.eye(n_agents, device=adjacency.device) + adjacency * (
                torch.ones_like(adjacency)
                - torch.eye(n_agents, device=adjacency.device)
            )

        coefficients = []
        for head in range(self.num_heads):
            source = torch.mm(hidden[:, head, :], self.a_i[head])
            target = torch.mm(hidden[:, head, :], self.a_j[head])
            coefficients.append(
                (
                    source.expand(n_agents, n_agents)
                    + target.transpose(0, 1).expand(n_agents, n_agents)
                ).unsqueeze(-1)
            )
        scores = self.leakyrelu(torch.cat(coefficients, dim=-1))
        mask = adjacency.unsqueeze(-1).expand_as(scores)
        attention = F.softmax(scores.masked_fill(mask <= 0, -1e9), dim=1) * mask
        if self.normalize:
            attention = attention / attention.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-15)
            attention = attention * mask
        dropped = F.dropout(attention, self.dropout, training=self.training)
        outputs = [
            torch.matmul(dropped[:, :, head], hidden[:, head, :])
            for head in range(self.num_heads)
        ]
        output = (
            torch.mean(torch.stack(outputs, dim=-1), dim=-1)
            if self.average
            else torch.cat(outputs, dim=-1)
        )
        if self.bias is not None:
            output = output + self.bias
        return output, attention


class HyperGraph(nn.Module):


    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.n_agents = args.nagents
        self.capability_aware = bool(args.capability_aware)
        self.capability_weight = float(args.capability_weight)
        self.hypergraph_temperature = float(args.hypergraph_temperature)
        self.hyperedge_size = int(args.hyperedge_size)
        self.task_observation_width = int(args.task_observation_width)
        self.task_observation_cells = self.task_observation_width ** 2
        task_center = self.task_observation_width // 2
        self.register_buffer(
            "_task_cell_distance",
            torch.tensor(
                [
                    abs(row - task_center) + abs(col - task_center)
                    for row in range(self.task_observation_width)
                    for col in range(self.task_observation_width)
                ],
                dtype=torch.long,
            ).view(1, 1, -1),
            persistent=False,
        )
        self.refinement_operator = getattr(
            args,
            "refinement_operator",
            "anchored_wave" if bool(args.use_wave) else "none",
        )
        valid_refinement_operators = {
            "none",
            "anchored_wave",
            "plain_diffusion",
            "repeated_gat",
        }
        if self.refinement_operator not in valid_refinement_operators:
            raise ValueError(
                "Unknown hypergraph refinement operator: "
                f"{self.refinement_operator!r}. Expected one of "
                f"{sorted(valid_refinement_operators)}."
            )
        # ``use_wave`` is retained in configs and historical checkpoint audits.
        # It now means that a refinement stage is active; ``refinement_operator``
        # records which controlled operator is used.
        self.use_wave = self.refinement_operator != "none"
        self.wave_iterations = int(args.wave_iterations)
        self.wave_tau = float(args.wave_tau)
        self.wave_damping = float(args.wave_damping)
        self.agent_levels = None

        if not 0.0 <= self.capability_weight <= 1.0:
            raise ValueError("capability_weight must be in [0, 1].")
        if self.hypergraph_temperature <= 0:
            raise ValueError("hypergraph_temperature must be positive.")
        self._validate_refinement_parameters()

        if self.refinement_operator == "repeated_gat":
            if int(args.gat_num_heads) * int(args.gat_hid_size) != int(args.hid_size):
                raise ValueError(
                    "Repeated-GAT control requires gat_num_heads * gat_hid_size "
                    "to equal hid_size."
                )
            self.refinement_gat = GAT(
                args.hid_size,
                args.gat_hid_size,
                dropout=0.0,
                negative_slope=0.2,
                num_heads=args.gat_num_heads,
                self_loop_type=1,
                average=False,
                normalize=True,
            )

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)
        self.query = nn.Linear(args.obs_size, args.qk_hid_size)
        if self.capability_aware:
            self.level_embedding = nn.Embedding(
                int(args.max_player_level) + 2, args.hid_size
            )
            self.capability_mlp = nn.Sequential(
                nn.Linear(3 * args.hid_size, args.hid_size),
                nn.ReLU(),
                nn.Linear(args.hid_size, 1),
            )

        self.node_to_edge = nn.Sequential(
            nn.Linear(args.hid_size, 2 * args.hid_size),
            nn.ReLU(),
            nn.Linear(2 * args.hid_size, args.hid_size),
            nn.ReLU(),
        )
        self.edge_to_node = nn.Sequential(
            nn.Linear(2 * args.hid_size, 2 * args.hid_size),
            nn.ReLU(),
            nn.Linear(2 * args.hid_size, args.hid_size),
            nn.ReLU(),
        )
        self.node_edge_attention = nn.Sequential(
            nn.Linear(2 * args.hid_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.last_incidence = None
        self.last_affinity = None
        self.last_visible_task_requirement = None
        self.last_coalition_cardinality = None
        self.last_target_partner_capability = None
        self.last_wave_diagnostics = None
        self.collect_wave_diagnostics = False




        self.collect_spectral_diagnostics = False

    def _validate_refinement_parameters(self) -> None:
        if self.wave_iterations < 1:
            raise ValueError("wave_iterations must be at least one.")
        if (
            self.refinement_operator == "anchored_wave"
            and not 0.0 < self.wave_damping < 1.0
        ):
            raise ValueError(
                "Input-retention damping must satisfy 0 < damping < 1."
            )
        if (
            self.refinement_operator in {"anchored_wave", "plain_diffusion"}
            and not 0.0 < self.wave_tau <= 1.0
        ):
            raise ValueError(
                "Dissipative diffusion requires 0 < wave_tau <= 1."
            )

    @staticmethod
    def _row_standardize(scores: torch.Tensor) -> torch.Tensor:
        mean = scores.mean(dim=-1, keepdim=True)
        variance = scores.var(dim=-1, keepdim=True, unbiased=False)
        return (scores - mean) / torch.sqrt(variance + 1e-8)

    def _capability_affinity(self, batch: int, n_agents: int) -> torch.Tensor:
        if self.agent_levels is None:
            raise RuntimeError(
                "Capability-aware construction requires the public fleet capability roster."
            )
        levels = self.agent_levels
        if levels.dim() == 1:
            levels = levels.unsqueeze(0)
        if levels.shape != (batch, n_agents):
            raise ValueError(
                f"Expected capability tensor {(batch, n_agents)}, got {tuple(levels.shape)}."
            )
        levels = levels.clamp(0, self.level_embedding.num_embeddings - 1)
        embedded = self.level_embedding(levels)
        left = embedded.unsqueeze(2).expand(-1, -1, n_agents, -1)
        right = embedded.unsqueeze(1).expand(-1, n_agents, -1, -1)
        symmetric_pair = torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=-1
        )
        return self.capability_mlp(symmetric_pair).squeeze(-1)

    def coalition_size(self, n_agents: int) -> int:

        if self.hyperedge_size > 0:
            return min(max(2, self.hyperedge_size), n_agents)
        return min(2, n_agents)

    def _nearest_visible_requirement(
        self, observations: torch.Tensor
    ) -> torch.Tensor:

        if observations.shape[-1] < self.task_observation_cells:
            raise ValueError(
                "Observation is shorter than the registered local task crop: "
                f"{observations.shape[-1]} < {self.task_observation_cells}."
            )
        local_tasks = observations[..., : self.task_observation_cells]
        width = self.task_observation_width
        distance = self._task_cell_distance
        visible = local_tasks > 0
        sentinel = width * width + 1
        nearest_distance = torch.where(
            visible, distance, torch.full_like(distance, sentinel)
        ).min(dim=-1, keepdim=True).values
        nearest = visible & (distance == nearest_distance)
        return torch.where(
            nearest, local_tasks, torch.zeros_like(local_tasks)
        ).max(dim=-1).values

    def _task_conditioned_cardinality(
        self, requirements: torch.Tensor, n_agents: int
    ) -> torch.Tensor:

        if self.hyperedge_size > 0:
            return torch.full_like(
                requirements,
                min(max(2, self.hyperedge_size), n_agents),
                dtype=torch.long,
            )
        if not self.capability_aware:
            return torch.full_like(
                requirements,
                min(max(2, n_agents // 4), n_agents),
                dtype=torch.long,
            )
        if self.agent_levels is None:
            raise RuntimeError(
                "Task-conditioned cardinality requires the public fleet capability roster."
            )
        levels = self.agent_levels
        if levels.dim() == 1:
            levels = levels.unsqueeze(0)
        if levels.shape[0] == 1 and requirements.shape[0] > 1:
            levels = levels.expand(requirements.shape[0], -1)
        if levels.shape != (requirements.shape[0], n_agents):
            raise ValueError(
                "Capability roster and observation batch disagree: "
                f"{tuple(levels.shape)} versus {(requirements.shape[0], n_agents)}."
            )
        numeric_levels = levels.to(requirements.dtype)
        anchor_capacity = numeric_levels.unsqueeze(-1)
        candidate_levels = numeric_levels.unsqueeze(1).expand(-1, n_agents, -1)
        diagonal = torch.eye(
            n_agents, device=requirements.device, dtype=torch.bool
        ).unsqueeze(0)
        candidate_levels = candidate_levels.masked_fill(diagonal, float("-inf"))
        sorted_candidates = torch.sort(
            candidate_levels, dim=-1, descending=True
        ).values[..., : max(0, n_agents - 1)]
        feasible_prefix_capacity = anchor_capacity + sorted_candidates.cumsum(dim=-1)
        sufficient = feasible_prefix_capacity >= requirements.unsqueeze(-1)
        first_sufficient = sufficient.to(torch.long).argmax(dim=-1) + 2
        impossible = ~sufficient.any(dim=-1)
        cardinality = torch.where(
            impossible,
            torch.full_like(first_sufficient, n_agents),
            first_sufficient,
        )
        cardinality = torch.where(
            requirements > 0,
            cardinality,
            torch.full_like(cardinality, self.coalition_size(n_agents)),
        )
        return cardinality.clamp(min=min(2, n_agents), max=n_agents)

    def _threshold_conditioned_member_scores(
        self,
        learned_scores: torch.Tensor,
        requirements: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:

        if self.agent_levels is None:
            raise RuntimeError(
                "Threshold-conditioned construction requires the public fleet capability roster."
            )
        batch, n_agents, _ = learned_scores.shape
        levels = self.agent_levels
        if levels.dim() == 1:
            levels = levels.unsqueeze(0)
        if levels.shape[0] == 1 and batch > 1:
            levels = levels.expand(batch, -1)
        if levels.shape != (batch, n_agents):
            raise ValueError(
                "Capability roster and affinity batch disagree: "
                f"{tuple(levels.shape)} versus {(batch, n_agents)}."
            )

        numeric_levels = levels.to(learned_scores.dtype)
        anchor_levels = numeric_levels
        candidate_levels = numeric_levels.unsqueeze(1).expand(-1, n_agents, -1)
        remaining_slots = (group_sizes - 1).clamp(min=1).to(learned_scores.dtype)
        remaining_demand = (requirements - anchor_levels).clamp(min=0.0)
        target_share = remaining_demand / remaining_slots
        max_level = numeric_levels.max(dim=-1, keepdim=True).values.clamp(min=1.0)
        normalized_gap = torch.abs(
            candidate_levels - target_share.unsqueeze(-1)
        ) / max_level.unsqueeze(-1)




        scale = max_level.unsqueeze(-1).reciprocal()
        feasibility_tie_break = (
            candidate_levels >= target_share.unsqueeze(-1)
        ).to(learned_scores.dtype) * (0.20 * scale)
        learned_tie_break = 0.10 * scale * torch.tanh(learned_scores)
        threshold_scores = (
            -normalized_gap + feasibility_tie_break + learned_tie_break
        )
        selection_scores = torch.where(
            (requirements > 0).unsqueeze(-1),
            threshold_scores,
            learned_scores,
        )
        self.last_target_partner_capability = target_share.detach()
        return selection_scores

    def build_incidence(
        self, observations: torch.Tensor, fixed_incidence=None
    ) -> torch.Tensor:

        batch, n_agents, _ = observations.shape
        if fixed_incidence is not None:
            incidence = torch.as_tensor(
                fixed_incidence,
                device=observations.device,
                dtype=observations.dtype,
            )
            if incidence.dim() == 2:
                incidence = incidence.unsqueeze(0)
            if incidence.shape[0] == 1 and batch > 1:
                incidence = incidence.expand(batch, -1, -1)
            expected = (batch, n_agents, n_agents)
            if tuple(incidence.shape) != expected:
                raise ValueError(
                    "Registered incidence must have shape "
                    f"{expected}, got {tuple(incidence.shape)}."
                )
            if torch.any((incidence != 0) & (incidence != 1)):
                raise ValueError("Registered incidence must be binary.")
            if torch.any(incidence.sum(dim=2) < 1) or torch.any(
                incidence.sum(dim=1) < 1
            ):
                raise ValueError(
                    "Registered incidence must cover every edge and agent."
                )
            visible_requirements = self._nearest_visible_requirement(observations)
            self.last_incidence = incidence.detach()
            self.last_affinity = torch.zeros(
                batch, n_agents, n_agents,
                device=observations.device,
                dtype=observations.dtype,
            )
            self.last_visible_task_requirement = visible_requirements.detach()
            self.last_coalition_cardinality = incidence.sum(dim=2).to(
                torch.long
            ).detach()
            self.last_target_partner_capability = torch.zeros_like(
                visible_requirements
            ).detach()
            return incidence
        query = self.query(observations)
        observation_affinity = torch.matmul(query, query.transpose(1, 2))
        observation_affinity = observation_affinity / math.sqrt(query.shape[-1])
        fused = self._row_standardize(observation_affinity)
        if self.capability_aware:
            capability = self._row_standardize(
                self._capability_affinity(batch, n_agents)
            )
            fused = (
                (1.0 - self.capability_weight) * fused
                + self.capability_weight * capability
            )

        visible_requirements = self._nearest_visible_requirement(observations)
        group_sizes = self._task_conditioned_cardinality(
            visible_requirements, n_agents
        )
        selection_scores = (
            self._threshold_conditioned_member_scores(
                fused, visible_requirements, group_sizes
            )
            if self.capability_aware
            else fused
        )
        hard = torch.eye(
            n_agents, device=observations.device, dtype=observations.dtype
        ).unsqueeze(0).expand(batch, -1, -1).clone()
        if n_agents > 1:
            diagonal = torch.eye(
                n_agents, device=observations.device, dtype=torch.bool
            ).unsqueeze(0)
            candidates = selection_scores.masked_fill(diagonal, float("-inf"))
            members = torch.argsort(
                candidates, dim=-1, descending=True, stable=True
            )
            member_rank = torch.arange(
                n_agents, device=observations.device
            ).view(1, 1, -1)
            selected = member_rank < (group_sizes - 1).unsqueeze(-1)
            selected_members = torch.zeros_like(hard)
            selected_members.scatter_(2, members, selected.to(hard.dtype))
            hard = torch.maximum(hard, selected_members)

        soft = torch.softmax(
            selection_scores / self.hypergraph_temperature, dim=-1
        )
        soft = soft * group_sizes.to(soft.dtype).unsqueeze(-1)
        incidence = hard + soft - soft.detach()
        self.last_incidence = hard.detach()
        self.last_affinity = selection_scores.detach()
        self.last_visible_task_requirement = visible_requirements.detach()
        self.last_coalition_cardinality = group_sizes.detach()
        return incidence

    def aggregate(self, observations: torch.Tensor, incidence: torch.Tensor):

        nodes = F.relu(self.obs_encoder(observations))
        transformed = self.node_to_edge(nodes)
        edge_degree = incidence.sum(dim=2, keepdim=True).clamp(min=1e-8)
        initial_edges = torch.matmul(incidence, transformed) / edge_degree

        _, n_edges, _ = initial_edges.shape
        n_nodes = transformed.shape[1]
        node_rep = transformed.unsqueeze(1).expand(-1, n_edges, -1, -1)
        edge_rep = initial_edges.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        logits = self.node_edge_attention(
            torch.cat([node_rep, edge_rep], dim=-1)
        ).squeeze(-1)
        hard_support = self.last_incidence > 0.5
        weights = torch.softmax(logits.masked_fill(~hard_support, -1e9), dim=2)
        weights = weights * incidence
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        edges = torch.matmul(weights, transformed)

        node_degree = incidence.sum(dim=1).unsqueeze(-1).clamp(min=1e-8)
        coalition_context = torch.matmul(incidence.transpose(1, 2), edges)
        coalition_context = coalition_context / node_degree
        coalitions = self.edge_to_node(torch.cat([coalition_context, nodes], dim=-1))
        return coalitions, edges

    @staticmethod
    def normalized_hypergraph_adjacency(
        node_by_edge: torch.Tensor,
    ) -> torch.Tensor:

        node_degree = node_by_edge.sum(dim=-1).clamp(min=1e-8)
        edge_degree = node_by_edge.sum(dim=-2).clamp(min=1e-8)
        left = torch.diag_embed(node_degree.rsqrt())
        edge_inverse = torch.diag_embed(edge_degree.reciprocal())
        adjacency = (
            left
            @ node_by_edge
            @ edge_inverse
            @ node_by_edge.transpose(-2, -1)
            @ left
        )
        return 0.5 * (adjacency + adjacency.transpose(-2, -1))

    @staticmethod
    def normalized_hypergraph_apply(
        node_by_edge: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:

        node_degree = node_by_edge.sum(dim=-1).clamp(min=1e-8)
        edge_degree = node_by_edge.sum(dim=-2).clamp(min=1e-8)
        node_scale = node_degree.rsqrt().unsqueeze(-1)
        scaled_nodes = node_scale * features
        edge_features = torch.matmul(
            node_by_edge.transpose(-2, -1), scaled_nodes
        )
        edge_features = edge_features * edge_degree.reciprocal().unsqueeze(-1)
        return node_scale * torch.matmul(node_by_edge, edge_features)

    def refine_features(
        self, coalition_features: torch.Tensor, incidence: torch.Tensor
    ) -> torch.Tensor:

        if self.refinement_operator == "none":
            self.last_wave_diagnostics = None
            return coalition_features

        node_by_edge = incidence.transpose(1, 2)

        if self.refinement_operator == "repeated_gat":
            adjacency = self.normalized_hypergraph_adjacency(node_by_edge)
            support = (adjacency > 0).to(dtype=coalition_features.dtype)
            current = coalition_features
            for _ in range(self.wave_iterations):
                batches = []
                for batch_index in range(current.shape[0]):
                    following, _attention = self.refinement_gat(
                        current[batch_index], support[batch_index]
                    )
                    batches.append(F.elu(following))
                current = torch.stack(batches, dim=0)
            self.last_wave_diagnostics = None
            return current

        node_scale = node_by_edge.sum(dim=-1).clamp(min=1e-8).rsqrt().unsqueeze(-1)
        edge_scale = (
            node_by_edge.sum(dim=-2).clamp(min=1e-8).reciprocal().unsqueeze(-1)
        )

        def apply_adjacency(features: torch.Tensor) -> torch.Tensor:
            scaled_nodes = node_scale * features
            edge_features = torch.matmul(
                node_by_edge.transpose(-2, -1), scaled_nodes
            )
            return node_scale * torch.matmul(
                node_by_edge, edge_scale * edge_features
            )

        initial = coalition_features
        current = coalition_features
        trajectory_norms = []
        for _ in range(self.wave_iterations):
            force = current - apply_adjacency(current)
            if self.refinement_operator == "anchored_wave":
                following = (
                    (1.0 - self.wave_damping) * initial
                    + self.wave_damping * (current - self.wave_tau * force)
                )
            elif self.refinement_operator == "plain_diffusion":
                following = current - self.wave_tau * force
            else:  # Guarded in __init__; retained as a defensive invariant.
                raise RuntimeError(
                    f"Unsupported refinement operator {self.refinement_operator!r}."
                )
            if self.collect_wave_diagnostics:
                trajectory_norms.append(
                    torch.linalg.vector_norm(
                        (following - current).detach(), dim=(-2, -1)
                    )
                )
            current = following

        if not self.collect_wave_diagnostics:
            self.last_wave_diagnostics = None
            return current

        with torch.no_grad():
            detached_incidence = node_by_edge.detach()
            initial_force = coalition_features.detach() - self.normalized_hypergraph_apply(
                detached_incidence, coalition_features.detach()
            )
            initial_energy = (
                coalition_features.detach() * initial_force
            ).sum(dim=(-2, -1)).clamp(min=0.0)
            batch = node_by_edge.shape[0]
            if self.collect_spectral_diagnostics:
                adjacency = self.normalized_hypergraph_adjacency(
                    detached_incidence
                )
                identity = torch.eye(
                    adjacency.shape[-1],
                    device=adjacency.device,
                    dtype=adjacency.dtype,
                ).unsqueeze(0)
                laplacian = identity - adjacency
                eigenvalues = torch.linalg.eigvalsh(laplacian.detach())
                lambda_min = eigenvalues.min(dim=-1).values
                lambda_max = eigenvalues.max(dim=-1).values
                lambda_max_is_bound = torch.zeros_like(lambda_max)
            else:


                lambda_min = torch.zeros(batch, device=node_by_edge.device)
                lambda_max = torch.ones(batch, device=node_by_edge.device)
                lambda_max_is_bound = torch.ones_like(lambda_max)
            final_force = current.detach() - self.normalized_hypergraph_apply(
                detached_incidence, current.detach()
            )
            final_energy = (current.detach() * final_force).sum(
                dim=(-2, -1)
            ).clamp(min=0.0)
            energy_ratio = torch.where(
                initial_energy > 1e-12,
                final_energy / initial_energy.clamp(min=1e-12),
                torch.zeros_like(final_energy),
            )
            self.last_wave_diagnostics = {
                "lambda_min": lambda_min.cpu(),
                "lambda_max": lambda_max.cpu(),
                "lambda_max_is_bound": lambda_max_is_bound.cpu(),
                "contraction_bound": torch.full_like(
                    lambda_max, self.wave_damping
                ).cpu(),
                "stability_margin": torch.full_like(
                    lambda_max, 1.0 - self.wave_damping
                ).cpu(),
                "initial_dirichlet_energy": initial_energy.cpu(),
                "final_dirichlet_energy": final_energy.cpu(),
                "energy_ratio": energy_ratio.cpu(),
                "update_norms": torch.stack(trajectory_norms, dim=1).cpu(),
            }
        return current

    def forward(self, observations: torch.Tensor, fixed_incidence=None):
        incidence = self.build_incidence(
            observations, fixed_incidence=fixed_incidence
        )
        coalition_features, _ = self.aggregate(observations, incidence)
        refined = self.refine_features(coalition_features, incidence)
        return refined, incidence

    def get_diagnostics(self) -> dict:
        return {
            "incidence": self.last_incidence,
            "affinity": self.last_affinity,
            "visible_task_requirement": self.last_visible_task_requirement,
            "coalition_cardinality": self.last_coalition_cardinality,
            "target_partner_capability": self.last_target_partner_capability,
            "wave": self.last_wave_diagnostics,
        }
