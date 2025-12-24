import time

class Task:
    def __init__(self, need: int, name:str="Unnamed"):
        self.need = need
        self.running = False
        self.name = name
        self.allocated = []
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.returncode: int | None = None

    def start(self, resources: list[int]):
        assert not self.running
        assert len(resources) >= self.need
        self.running = True
        self.allocated = resources.copy()
        self.start_time = time.time()

    def check_finish(self) -> bool:
        self.returncode = 0
        return True
        raise NotImplementedError()

    def close(self):
        self.running = False
        self.end_time = time.time()

class Scheduler:
    def __init__(self, resources: list[int], tasks: list[Task]):
        self.resources = resources.copy()
        self.tasks = tasks.copy()
        self.running_tasks:list[Task] = []
        self.succeeded_tasks: list[Task] = []
        self.failed_tasks: list[Task] = []
        self.tasks.sort(key=lambda x: x.need, reverse=True)

    def _check_runnings(self):
        finished = []
        for t in self.running_tasks:
            if t.check_finish():
                finished.append(t)
        for t in finished:
            self.running_tasks.remove(t)
            self.resources.extend(t.allocated)
            t.close()
            if t.returncode == 0:
                self.succeeded_tasks.append(t)
            else:
                self.failed_tasks.append(t)

    def do_schedule(self):
        self._check_runnings()
        while self.resources and self.tasks:
            for t in self.tasks:
                if t.need <= len(self.resources):
                    alloc = self.resources[: t.need]
                    self.resources = self.resources[t.need :]
                    t.start(alloc)
                    self.running_tasks.append(t)
                    self.tasks.remove(t)
                    break
            else:
                break
            
    def finished(self):
        return not self.tasks and not self.running_tasks

    def run(self):
        while not self.finished():
            self.do_schedule()
            time.sleep(1)