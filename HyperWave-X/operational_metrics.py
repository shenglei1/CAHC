def write_comparison_csv(directory, rows):

    import csv
    from pathlib import Path
    identity = ['Method', 'Scenario', 'Train_Seed', 'Eval_Seed', 'Episode']
    performance = ['Episode_Steps', 'Agent_Count',
                   'Workload_Total', 'Workload_Completed',
                   'Capacity_Workload_Total', 'Capacity_Workload_Completed',
                   'Capacity_Completion_Ratio', 'Task_Completion_Ratio',
                   'Episode_Success', 'Episode_Success_Rate', 'Throughput',
                   'Collisions', 'Collision_Rate',
                   'Load_Attempts', 'False_Load_Attempts',
                   'Missing_Member_Rate', 'Ready_Task_Load_Attempts',
                   'Missing_Member_Task_Attempts', 'Invalid_Load_Actions',
                   'Synchronization_Success_Rate', 'Synchronized_Load_Successes',
                   'Mean_Released_Backlog', 'Completed_Task_Delay_Sum',
                   'Completed_Task_Delay_Count', 'Unfinished_Task_Count',
                   'Completed_Task_Excess_Capacity_Sum',
                   'First_Completion_Step_Or_Horizon', 'All_Completed_Step_Or_Horizon']
    fields = identity + performance
    with (Path(directory) / 'comparison_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

class OperationalMetrics:
    def __init__(self, raw):
        self.attempts = self.failed = self.missing = self.ready_attempts = 0
        self.invalid = self.successes = self.backlog_area = 0
        self.completion = {}
        self.excess = []

    def before_step(self, raw, actions):

        groups = {}
        for player, action in zip(raw.players, actions):
            if action != getattr(raw, "load_action", 5):
                continue
            position = raw._adjacent_task_location(player.position)
            if position is None:
                self.invalid += 1
            else:
                groups.setdefault(position, []).append(player)
        self.pending = []
        for position, loaders in groups.items():
            task = raw._active_task_by_position[position]
            requirement = int(task['requirement'])
            capacity = sum(p.level for p in loaders)

            ready_capacity = sum(
                p.level for p in raw.players
                if raw._adjacent_task_location(p.position) == position
            )
            self.attempts += 1
            eligible = task.get("eligible", True)
            self.failed += int(not eligible or capacity < requirement)
            self.ready_attempts += int(eligible and ready_capacity >= requirement)
            self.missing += int(eligible and ready_capacity >= requirement and capacity < requirement)
            self.pending.append((task, capacity - requirement))

    def after_step(self, raw):
        for task, excess in self.pending:
            if task['completed']:
                self.successes += 1
                self.excess.append(excess)
        for task in raw._task_plan:
            if task['completed'] and task['task_id'] not in self.completion:
                self.completion[task['task_id']] = raw.current_step
        self.backlog_area += sum(t['released'] and not t['completed'] and not t.get('expired', False)
                                 for t in raw._task_plan)

    def summary(self, raw):
        delays = [
            self.completion[t['task_id']] - t['release_after_step']
            for t in raw._task_plan if t['task_id'] in self.completion
        ]
        return {
            'Load_Attempts': self.attempts,
            'False_Load_Attempts': self.failed,
            'Missing_Member_Rate': self.missing / self.ready_attempts if self.ready_attempts else 0.0,
            'Ready_Task_Load_Attempts': self.ready_attempts,
            'Missing_Member_Task_Attempts': self.missing,
            'Invalid_Load_Actions': self.invalid,
            'Synchronization_Success_Rate': self.successes / self.attempts if self.attempts else 0.0,
            'Synchronized_Load_Successes': self.successes,
            'Mean_Released_Backlog': self.backlog_area / raw.current_step,
            'Completed_Task_Delay_Sum': sum(delays),
            'Completed_Task_Delay_Count': len(delays),
            'Unfinished_Task_Count': len(raw._task_plan) - len(delays),
            'Completed_Task_Excess_Capacity_Sum': sum(self.excess),
            'First_Completion_Step_Or_Horizon': min(self.completion.values(), default=raw._max_episode_steps),
            'All_Completed_Step_Or_Horizon': max(self.completion.values()) if len(delays) == len(raw._task_plan) else raw._max_episode_steps,
        }
