"""Evaluation-only logical aggregation counts.

The counters describe model-level feature transfers, not radio packets or
serialized network traffic. They never update model parameters.
"""
import csv
from pathlib import Path


class FairCosts:
    def __init__(self, model, method):
        self.model, self.method = model, method
        self.handles, self.wrappers = [], []
        self.calls = {}
        self.steps = self.transfers = self.scalars = self.incidence_transfers = 0
        self.rows = []
        for _, module in model.named_modules():
            kind = type(module).__name__
            if kind in {'GAT', 'GraphAttention', 'GCNConv', 'MultiheadAttention', 'HyperGraph'}:
                self.handles.append(module.register_forward_hook(self._aggregation))
        self._capture(model, 'forward', 'policy')

    def _capture(self, obj, attr, name):
        original = getattr(obj, attr)
        self.calls[name] = 0

        def wrapped(*args, **kwargs):
            self.calls[name] += 1
            return original(*args, **kwargs)

        setattr(obj, attr, wrapped)
        self.wrappers.append((obj, attr, original))

    def _add(self, count, width):
        self.transfers += count
        self.scalars += count * width

    def _aggregation(self, module, args, output):
        kind = type(module).__name__
        if kind == 'HyperGraph':
            incidence = output[-1].detach()
            passes = 2 + (2 * module.wave_iterations if getattr(module, 'use_wave', False) else 0)
            self.incidence_transfers += int((incidence > .5).sum().item()) * passes
        elif kind in {'GAT', 'GraphAttention'}:
            features, adjacency = args[:2]
            adjacency = adjacency.detach().clone()
            adjacency.fill_diagonal_(0)
            self._add(int((adjacency != 0).sum().item()), features.shape[-1])
        elif kind == 'GCNConv':
            features, edges = args[:2]
            self._add(int((edges[0] != edges[1]).sum().item()), features.shape[-1])
        else:
            query, key, value = args[:3]
            axis = 1 if module.batch_first else 0
            n, m = query.shape[axis], key.shape[axis]
            self._add(n * m - min(n, m), value.shape[-1])

    def episode(self, identity, steps):
        self.steps += steps
        row = {key: identity[key] for key in ['Method', 'Scenario', 'Train_Seed',
                                             'Eval_Seed', 'Episode']}
        previous = self.rows[-1] if self.rows else {}
        for name, total in [('Logical_Transfers', self.transfers),
                            ('Logical_Feature_Scalars', self.scalars),
                            ('Incidence_Traversals', self.incidence_transfers)]:
            prior = previous.get('_' + name, 0)
            row[name + '_Per_Step'] = (total - prior) / steps
            row['_' + name] = total
        row['Logical_Feature_Bytes_FP32_Per_Step'] = row['Logical_Feature_Scalars_Per_Step'] * 4
        self.rows.append(row)

    def finish(self, directory):
        for handle in self.handles:
            handle.remove()
        for obj, attr, original in self.wrappers:
            setattr(obj, attr, original)
        clean_rows = [{k: v for k, v in row.items() if not k.startswith('_')} for row in self.rows]
        with (Path(directory) / 'communication_costs.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(clean_rows[0]))
            writer.writeheader()
            writer.writerows(clean_rows)
