from lazyexp import exenv, exper

envs = exper.loadEnvs("test_exp")
# exenv.pack_envs(envs, dry_run=False)
exenv.move_envs(envs, "outputs", "outputs_moved")