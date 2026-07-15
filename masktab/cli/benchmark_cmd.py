import click
import sys
from masktab.training.benchmark import run_benchmark

@click.command()
@click.option('--encoder-version-id', type=int, required=True, help='ID of the pretrained encoder version')
@click.option('--csv', 'csv_path', required=True, type=click.Path(exists=True), help='Path to the input CSV file')
@click.option('--db', 'db_path', required=True, type=click.Path(), help='Path to the SQLite database')
@click.option('--target-column', required=True, type=str, help='Name of the target column')
@click.option('--task-type', required=True, type=click.Choice(['classification', 'regression']), help='Type of task')
@click.option('--labeled-fraction-partial', type=float, default=0.2, help='Fraction of data for partial supervised training')
@click.option('--feature-columns', type=str, help='Comma-separated list of feature columns (optional)')
@click.option('--n-classes', type=int, help='Number of classes for classification (optional)')
@click.option('--epochs', type=int, default=100, help='Number of training epochs')
@click.option('--batch-size', type=int, default=32, help='Training batch size')
@click.option('--seed', 'random_seed', type=int, default=42, help='Random seed')
def benchmark(encoder_version_id, csv_path, db_path, target_column, task_type, labeled_fraction_partial, feature_columns, n_classes, epochs, batch_size, random_seed):
    """Run a benchmark evaluating multiple configurations."""
    if feature_columns:
        feature_columns = [c.strip() for c in feature_columns.split(",")]
        
    try:
        result = run_benchmark(
            encoder_version_id=encoder_version_id,
            csv_path=csv_path,
            db_path=db_path,
            target_column=target_column,
            task_type=task_type,
            feature_columns=feature_columns,
            n_classes=n_classes,
            labeled_fraction_partial=labeled_fraction_partial,
            random_seed=random_seed,
            epochs=epochs,
            batch_size=batch_size
        )
        
        click.echo("--- Benchmark Summary ---")
        
        for arm in ["pretrained_full", "random_encoder_full", "supervised_full", "supervised_partial"]:
            arm_res = result[arm]
            click.echo(f"\nArm: {arm}")
            click.echo(f"  Training Samples: {arm_res['n_training_samples']}")
            click.echo(f"  {arm_res['val_metric_name']}: {arm_res['val_metric_value']:.6f}")
            
        click.echo("\n--- Comparisons ---")
        click.echo(f"Pretrained within 5% of Supervised Full: {result['pretrained_within_5pct_of_supervised']}")
        click.echo(f"Pretrained beats Supervised Partial: {result['pretrained_beats_partial_supervised']}")
        
    except Exception as e:
        click.echo(f"Error during benchmark: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    benchmark()
