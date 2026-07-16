import click
import sys
from masktab.validation.synthetic_drift import run_drift_validation
from masktab.validation.latency_bench import benchmark_inference_latency

@click.command('validate-drift')
@click.pass_context
@click.option('--task-name', required=True, type=str, help='Task name')
@click.option('--csv', 'holdout_csv_path', required=True, type=click.Path(exists=True), help='Holdout CSV path')
@click.option('--features', 'features_to_shift', required=True, type=str, help='Comma-separated features to shift')
@click.option('--scale-factor', type=float, default=3.0, help='Scale factor')
def validate_drift_cmd(ctx, task_name, holdout_csv_path, features_to_shift, scale_factor):
    """Run synthetic drift validation."""
    db_path = ctx.obj.get('db_path')
    features = [f.strip() for f in features_to_shift.split(",")]
    try:
        res = run_drift_validation(task_name, db_path, holdout_csv_path, features, scale_factor)
        click.echo("--- Drift Validation Summary ---")
        click.echo(f"Features Shifted: {features}")
        click.echo(f"Baseline Drift Detected: {res['baseline_drift_detected']}")
        click.echo(f"Shifted Drift Detected: {res['shifted_drift_detected']}")
        click.echo(f"Passed Validation: {res['passed_validation']}")
    except Exception as e:
        click.echo(f"Error during drift validation: {e}", err=True)
        sys.exit(1)

@click.command('bench-latency')
@click.pass_context
@click.option('--task-name', required=True, type=str, help='Task name')
@click.option('--csv', 'sample_csv_path', required=True, type=click.Path(exists=True), help='Sample CSV path')
@click.option('--batch-size', type=int, default=100, help='Batch size')
@click.option('--n-trials', type=int, default=10, help='Number of trials')
def bench_latency_cmd(ctx, task_name, sample_csv_path, batch_size, n_trials):
    """Benchmark inference latency."""
    db_path = ctx.obj.get('db_path')
    try:
        res = benchmark_inference_latency(task_name, db_path, sample_csv_path, batch_size, n_trials)
        click.echo("--- Latency Benchmark Summary ---")
        click.echo(f"Mean Latency: {res['mean_ms']:.2f}ms")
        click.echo(f"Median Latency: {res['median_ms']:.2f}ms")
        click.echo(f"P95 Latency: {res['p95_ms']:.2f}ms")
        click.echo(f"Max Latency: {res['max_ms']:.2f}ms")
        click.echo(f"Target: {res['target_ms']:.2f}ms")
        click.echo(f"Meets Target: {res['meets_target']}")
    except Exception as e:
        click.echo(f"Error during latency benchmark: {e}", err=True)
        sys.exit(1)
