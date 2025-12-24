from lazyexp.expui import SchedulerUI
from lazyexp.scheduler import Scheduler, Task

if __name__ == "__main__":
    # test code
    tasks = [Task(need=2), Task(need=1), Task(need=3)]
    scheduler = Scheduler(resources=[0, 1, 2, 3, 4], tasks=tasks)
    ui = SchedulerUI(scheduler)
    ui.run()
    
    tasks = [Task(need=2), Task(need=2), Task(need=2)]
    scheduler = Scheduler(resources=[0, 1, 2, 3, 4], tasks=tasks)
    ui = SchedulerUI(scheduler)
    ui.run()