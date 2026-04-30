from lazyexp import runners

@runners.cmd_runner
def cmd_maker(env):
    return ["python"]

print(cmd_maker.__name__)