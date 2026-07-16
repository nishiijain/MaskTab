import click
import os

from masktab.cli.pretrain_cmd import pretrain
from masktab.cli.finetune_cmd import finetune
from masktab.cli.benchmark_cmd import benchmark
from masktab.cli.infer_cmd import infer
from masktab.cli.rollback_cmd import rollback_head_cmd, rollback_encoder_cmd, list_versions_cmd
from masktab.cli.validation_cmd import validate_drift_cmd, bench_latency_cmd

@click.group()
@click.option('--db', 'db_path', type=click.Path(), default="masktab.db", help="Path to SQLite database")
@click.pass_context
def cli(ctx, db_path):
    """MaskTab CLI entrypoint."""
    ctx.ensure_object(dict)
    ctx.obj['db_path'] = os.path.abspath(db_path)

cli.add_command(pretrain)
cli.add_command(finetune)
cli.add_command(benchmark)
cli.add_command(infer)
cli.add_command(rollback_head_cmd, name="rollback-head")
cli.add_command(rollback_encoder_cmd, name="rollback-encoder")
cli.add_command(list_versions_cmd, name="list-versions")
cli.add_command(validate_drift_cmd, name="validate-drift")
cli.add_command(bench_latency_cmd, name="bench-latency")

if __name__ == "__main__":
    cli()
