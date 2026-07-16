import click
import sys
from masktab.training.pretrain import pretrain_encoder

@click.command()
@click.pass_context
@click.option('--csv', 'csv_path', required=True, type=click.Path(exists=True), help='Path to the input CSV file')
@click.option('--epochs', type=int, default=50, help='Number of training epochs')
@click.option('--batch-size', type=int, default=64, help='Training batch size')
@click.option('--lr', 'learning_rate', type=float, default=1e-3, help='Learning rate')
@click.option('--mask-prob', type=float, default=0.3, help='Probability of masking a feature')
@click.option('--alpha', type=float, default=0.5, help='Weight for numerical loss component')
@click.option('--val-split', type=float, default=0.15, help='Validation split ratio')
@click.option('--artifact-dir', type=click.Path(), default="artifacts/encoders", help='Directory to save artifacts')
@click.option('--seed', 'random_seed', type=int, default=42, help='Random seed for reproducibility')
@click.option('--patience', 'early_stopping_patience', type=int, default=8, help='Early stopping patience')
def pretrain(ctx, csv_path, epochs, batch_size, learning_rate, mask_prob, alpha, val_split, artifact_dir, random_seed, early_stopping_patience):
    """Pretrain the Masked Autoencoder."""
    db_path = ctx.obj.get('db_path')
    try:
        result = pretrain_encoder(
            csv_path=csv_path,
            db_path=db_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mask_prob=mask_prob,
            alpha=alpha,
            val_split=val_split,
            artifact_dir=artifact_dir,
            random_seed=random_seed,
            early_stopping_patience=early_stopping_patience
        )
        
        click.echo("--- Pretraining Summary ---")
        click.echo(f"Artifact Path: {result['artifact_path']}")
        click.echo(f"Encoder Version ID: {result['encoder_version_id']}")
        click.echo(f"Dataset Hash: {result['dataset_hash']}")
        click.echo(f"Epochs Run: {result['epochs_run']}")
        click.echo(f"Final Train Loss: {result['final_train_loss']:.6f}")
        click.echo(f"Best Val Loss: {result['best_val_loss']:.6f}")
        click.echo(f"Early Stopped: {result['early_stopped']}")
        
    except Exception as e:
        click.echo(f"Error during pretraining: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    pretrain()
