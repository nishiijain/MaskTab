import click
import sys
from masktab.training.finetune import finetune_head

@click.command()
@click.pass_context
@click.option('--encoder-version-id', type=int, required=True, help='ID of the pretrained encoder version')
@click.option('--csv', 'csv_path', required=True, type=click.Path(exists=True), help='Path to the input CSV file')
@click.option('--target-column', required=True, type=str, help='Name of the target column')
@click.option('--task-type', required=True, type=click.Choice(['classification', 'regression']), help='Type of task')
@click.option('--task-name', required=True, type=str, help='Name of the fine-tuning task')
@click.option('--feature-columns', type=str, help='Comma-separated list of feature columns (optional)')
@click.option('--n-classes', type=int, help='Number of classes for classification (optional)')
@click.option('--epochs', type=int, default=100, help='Number of training epochs')
@click.option('--batch-size', type=int, default=32, help='Training batch size')
@click.option('--lr', 'learning_rate', type=float, default=1e-3, help='Learning rate')
@click.option('--val-split', type=float, default=0.15, help='Validation split ratio')
@click.option('--artifact-dir', type=click.Path(), default="artifacts/heads", help='Directory to save artifacts')
@click.option('--seed', 'random_seed', type=int, default=42, help='Random seed')
@click.option('--patience', 'early_stopping_patience', type=int, default=10, help='Early stopping patience')
def finetune(ctx, encoder_version_id, csv_path, target_column, task_type, task_name, feature_columns, n_classes, epochs, batch_size, learning_rate, val_split, artifact_dir, random_seed, early_stopping_patience):
    """Fine-tune a task head on top of a frozen pretrained encoder."""
    db_path = ctx.obj.get('db_path')
    if feature_columns:
        feature_columns = [c.strip() for c in feature_columns.split(",")]
        
    try:
        result = finetune_head(
            encoder_version_id=encoder_version_id,
            csv_path=csv_path,
            db_path=db_path,
            target_column=target_column,
            task_type=task_type,
            feature_columns=feature_columns,
            n_classes=n_classes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_split=val_split,
            artifact_dir=artifact_dir,
            random_seed=random_seed,
            early_stopping_patience=early_stopping_patience,
            task_name=task_name
        )
        
        click.echo("--- Fine-tuning Summary ---")
        click.echo(f"Task Name: {result['task_name']}")
        click.echo(f"Head Version ID: {result['head_version_id']}")
        click.echo(f"Encoder Version ID: {result['encoder_version_id']}")
        click.echo(f"Artifact Path: {result['artifact_path']}")
        click.echo(f"Validation Metric ({result['val_metric_name']}): {result['val_metric_value']:.6f}")
        click.echo(f"Epochs Run: {result['epochs_run']}")
        click.echo(f"Early Stopped: {result['early_stopped']}")
        
    except Exception as e:
        click.echo(f"Error during fine-tuning: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    finetune()
