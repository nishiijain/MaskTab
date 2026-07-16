import click
import json
import sys
from masktab.registry.rollback import rollback_head, rollback_encoder, list_versions

@click.command('rollback-head')
@click.pass_context
@click.option('--task-name', required=True, type=str, help='Task name')
@click.option('--target-head-version-id', required=True, type=int, help='ID of head version to rollback to')
def rollback_head_cmd(ctx, task_name, target_head_version_id):
    """Roll back the active head for a task to a previous version."""
    db_path = ctx.obj.get('db_path')
    try:
        res = rollback_head(task_name, target_head_version_id, db_path)
        click.echo("--- Rollback Head Summary ---")
        click.echo(f"Previous Active ID: {res['previous_active_id']}")
        click.echo(f"New Active ID: {res['new_active_id']}")
    except Exception as e:
        click.echo(f"Error during rollback_head: {e}", err=True)
        sys.exit(1)

@click.command('rollback-encoder')
@click.pass_context
@click.option('--task-name', required=True, type=str, help='Task name')
@click.option('--target-encoder-version-id', required=True, type=int, help='ID of encoder version to rollback to')
def rollback_encoder_cmd(ctx, task_name, target_encoder_version_id):
    """
    Re-points the active head for a task to a different encoder version.
    Does NOT retrain the head; only repoints metadata.
    """
    db_path = ctx.obj.get('db_path')
    try:
        res = rollback_encoder(task_name, target_encoder_version_id, db_path)
        click.echo("--- Rollback Encoder Summary ---")
        click.echo(f"Previous Head ID: {res['previous_head_id']}")
        click.echo(f"New Head ID: {res['new_head_id']}")
        click.echo(f"Target Encoder Version: {res['target_encoder_version_id']}")
    except Exception as e:
        click.echo(f"Error during rollback_encoder: {e}", err=True)
        sys.exit(1)

@click.command('list-versions')
@click.pass_context
@click.option('--task-name', required=True, type=str, help='Task name')
def list_versions_cmd(ctx, task_name):
    """List version history for a task."""
    db_path = ctx.obj.get('db_path')
    try:
        res = list_versions(task_name, db_path)
        click.echo(f"--- Version History for {res['task_name']} ---")
        for v in res['history']:
            active_marker = "*" if v["is_active"] else " "
            click.echo(
                f"[{active_marker}] Head ID: {v['head_id']} | "
                f"Metric ({v['val_metric_name']}): {v['val_metric_value']:.4f} | "
                f"Encoder ID: {v['encoder_version_id']} | "
                f"Created: {v['head_created_at']}"
            )
    except Exception as e:
        click.echo(f"Error listing versions: {e}", err=True)
        sys.exit(1)
