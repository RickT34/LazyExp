import time
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from .scheduler import Scheduler, Task


# --- UI 构建函数 ---
class SchedulerUI:
    def __init__(self, s: Scheduler, title:str|None = None):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        self.layout = layout
        self.title = title
        self.scheduler = s
        self.total_tasks_count = len(s.tasks)
        self.console = Console()
        
    def generate_progress_table(self):
        table = Table(title="Tasks Running", expand=True)
        table.add_column("Name", justify="left")
        table.add_column("Resource", style="magenta")
        table.add_column("Time", justify="center")

        if self.scheduler.finished():
            for p in self.scheduler.succeeded_tasks:
                table.add_row(
                    f"{p.name}",
                    str(p.allocated),
                    f"[green]{p.end_time - p.start_time if p.start_time and p.end_time else 0.0:.1f} s[/green]",
                )
        else:
            for p in self.scheduler.running_tasks:
                
                table.add_row(
                    p.name,
                    str(p.allocated),
                    f"{time.time() - p.start_time:.1f}s" if p.start_time else "N/A",
                )
            for p in self.scheduler.tasks:
                table.add_row(
                    p.name,
                    "Wanting " + str(p.need),
                    "[yellow]Waiting[/yellow]",
                )
        return table


    def generate_resource_panel(self):
        idle = self.scheduler.resources
        content = f"[bold green]Available Resources:[/bold green] {len(idle)}\n"
        content += f"[blue]{', '.join(map(str, idle))}[/blue]"
        return Panel(content, title="Resource Pool Status", border_style="blue")


    def generate_failed_panel(self):
        content = "\n".join([f"[red]{t.name}[/red]" for t in self.scheduler.failed_tasks])
        return Panel(content or "No failed tasks", title="Failed Tasks", border_style="red")


    def generate_overall_progress(self):
        done = len(self.scheduler.succeeded_tasks+self.scheduler.failed_tasks)
        percentage = (done / self.total_tasks_count) * 100

        prog = Progress(
            TextColumn("[bold blue]Overall Progress"),
            BarColumn(bar_width=None),
            TextColumn(f"[progress.percentage]{percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = prog.add_task("total", total=self.total_tasks_count, completed=done)
        return Panel(prog, title="Overall Progress")

    def update(self):
        self.layout["left"].update(self.generate_progress_table())
        self.layout["right"].split_column(
            Layout(self.generate_resource_panel()),
            Layout(self.generate_failed_panel()),
        )
        self.layout["footer"].update(self.generate_overall_progress())

    def run(self):
        self.layout["header"].update(
            Panel(
                f"[bold white]🚀 TRs LazyExp Scheduler - {self.title}[/bold white]",
                style="on blue",
                title_align="center",
            )
        )

        with Live(self.layout, refresh_per_second=4, screen=False):
            while not self.scheduler.finished():
                self.scheduler.do_schedule()
                try:
                    self.update()
                except Exception as e:
                    print(f"Scheduler UI update error: {e}")
                time.sleep(2)
            self.update()

        

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