import random
import torch
import torch.nn.functional as F

class ICLData:
    def __init__(self, config, device):
        self.P = config.P
        self.ctx = config.ctx
        self.device = device

        all_rules = [(a, b) for a in range(self.P) for b in range(self.P)]
        all_inputs = [(x, y) for x in range(self.P) for y in range(self.P)]
        random.shuffle(all_rules)
        random.shuffle(all_inputs)

        n_rules = int(len(all_rules) * config.rule_frac)
        n_inputs = int(len(all_inputs) * config.input_frac)

        self.train_rules = all_rules[:n_rules]
        self.test_rules = all_rules[n_rules:]
        self.train_inputs = all_inputs[:n_inputs]
        self.test_inputs = all_inputs[n_inputs:]

        print(
            f"Rules: train {len(self.train_rules)} / test {len(self.test_rules)} | "
            f"Inputs: train {len(self.train_inputs)} / test {len(self.test_inputs)}"
        )

        assert len(self.train_inputs) >= self.ctx and len(self.test_inputs) >= self.ctx, (
            f"input pools too small (train={len(self.train_inputs)}, "
            f"test={len(self.test_inputs)}, need >= {self.ctx}). "
            f"Lower --input_frac (max ~{1 - self.ctx / len(all_inputs):.2f}) or --ctx."
        )
        assert len(self.test_rules) > 0, "rule_frac too large: no OOD rules left"

        self.rectangles = self.build_rectangles(self.train_rules)

        print("Train rectangles:", len(self.rectangles))
        assert self.rectangles, "no rectangle fits inside the training rules; raise --rule_frac"

        self.label_pos = list(range(2, 3 * self.ctx, 3))
        self.pred_pos = [p - 1 for p in self.label_pos]

        self.rect_tensor = torch.tensor(
            self.rectangles, dtype=torch.long, device=device
        )
        self.train_rules_t = torch.tensor(
            self.train_rules, dtype=torch.long, device=device
        )
        self.test_rules_t = torch.tensor(
            self.test_rules, dtype=torch.long, device=device
        )
        self.train_inputs_t = torch.tensor(
            self.train_inputs, dtype=torch.long, device=device
        )
        self.test_inputs_t = torch.tensor(
            self.test_inputs, dtype=torch.long, device=device
        )

    def build_rectangles(self, rules):
        rule_set = set(rules)
        rectangles = []
        keys = set()

        for a, b in rules:
            for da in range(1, self.P):
                a2 = (a + da) % self.P

                if (a2, b) not in rule_set:
                    continue

                for db in range(1, self.P):
                    b2 = (b + db) % self.P

                    if (a, b2) not in rule_set or (a2, b2) not in rule_set:
                        continue

                    rectangle = ((a, b), (a2, b), (a, b2), (a2, b2))
                    key = tuple(sorted(rectangle))

                    if key not in keys:
                        keys.add(key)
                        rectangles.append(rectangle)

        return rectangles

    def sample_inputs(self, input_pool, groups, generator=None):
        pool_size = input_pool.size(0)
        weights = torch.ones(groups, pool_size, device=self.device)

        indices = torch.multinomial(
            weights,
            self.ctx,
            replacement=False,
            generator=generator
        )

        return input_pool[indices]

    def build_sequences(self, rules, pairs):
        groups, rules_per_group, _ = rules.shape

        x = pairs[..., 0][:, None, :].expand(
            groups, rules_per_group, self.ctx
        )
        y = pairs[..., 1][:, None, :].expand(
            groups, rules_per_group, self.ctx
        )

        a = rules[..., 0][:, :, None]
        b = rules[..., 1][:, :, None]

        z = (a * x + b * y) % self.P

        return torch.stack([x, y, z], dim=-1).reshape(
            groups * rules_per_group,
            3 * self.ctx
        )

    def train_batch(self, groups):
        indices = torch.randint(
            0,
            self.rect_tensor.size(0),
            (groups,),
            device=self.device
        )

        rules = self.rect_tensor[indices]
        pairs = self.sample_inputs(self.train_inputs_t, groups)

        return self.build_sequences(rules, pairs)

    def loss(self, logits, ids):
        predictions = logits[:, self.pred_pos, :]
        targets = ids[:, self.label_pos]

        return F.cross_entropy(
            predictions.reshape(-1, predictions.size(-1)),
            targets.reshape(-1)
        )